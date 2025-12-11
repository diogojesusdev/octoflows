from dataclasses import dataclass
import nest_asyncio
import asyncio
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.client import PubSub
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

import src.storage.storage as storage
from src.utils.logger import create_logger

logger = create_logger(__name__)

nest_asyncio.apply()

@dataclass
class SubscriptionInfo:
    """Information about a subscription callback."""
    subscription_id: str
    callback: Callable[[dict, str], Any]
    decode_responses: bool
    worker_id: Optional[str]

class RedisStorage(storage.Storage):
    @dataclass
    class Config(storage.Storage.Config):
        address: tuple[str, int]
        password: str
        max_concurrent_ops: int = 20

        def create_instance(self) -> "RedisStorage":
            return RedisStorage(self)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.redis_config = config
        self._connection: Optional[Redis] = None
        self._pubsub: Optional[PubSub] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._sub_lock = asyncio.Lock()
        
        # Gatekeeper for all Redis operations
        self._op_semaphore = asyncio.Semaphore(self.redis_config.max_concurrent_ops)
        
        # self.ARTIFICIAL_NETWORK_LATENCY_S = 0.030 
        self.ARTIFICIAL_NETWORK_LATENCY_S = 0
        self._channel_subscriptions: Dict[str, Dict[str, SubscriptionInfo]] = {}
        self._channel_tasks: Dict[str, Tuple[asyncio.Task, Any]] = {}
        self._conn_lock = asyncio.Lock()

    async def _simulate_network_latency(self) -> None:
        if self.ARTIFICIAL_NETWORK_LATENCY_S > 0:
            await asyncio.sleep(self.ARTIFICIAL_NETWORK_LATENCY_S)

    async def _get_or_create_connection(self) -> Redis:
        async with self._conn_lock:
            if not self._connection:
                self._pool = ConnectionPool(
                    host=self.redis_config.address[0],
                    port=self.redis_config.address[1],
                    db=0,
                    password=self.redis_config.password,
                    decode_responses=False,
                    # Increased timeouts to handle queuing time if semaphore is busy
                    socket_connect_timeout=60,
                    socket_timeout=60,
                    retry_on_timeout=True,
                    health_check_interval=25,
                    socket_keepalive=True,
                )

                self._connection = Redis(
                    connection_pool=self._pool,
                    retry_on_error=[ConnectionError, TimeoutError, OSError, BufferError],
                    retry=Retry(backoff=ExponentialBackoff(), retries=5),
                )
            return self._connection

    async def get(self, key: str) -> Any:
        # Wrap the operation in the semaphore
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            if not await conn.exists(key):
                return None
            return await conn.get(key)

    async def set(self, key: str, value: Any) -> bool:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            return await conn.set(key, value)

    async def atomic_increment_and_get(self, key: str) -> int:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            return await conn.incr(key, amount=1)

    async def exists(self, *keys: str) -> int:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            return await conn.exists(*keys)
    
    async def close_connection(self):
        # Closing does not need the semaphore, it is cleanup
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task # type: ignore
            except asyncio.CancelledError:
                pass
            self._pubsub_task = None
        if self._pubsub:
            await self._pubsub.aclose()
            self._pubsub = None
        if self._connection:
            await self._connection.aclose()
            disconnect_from_pool = self._connection.connection_pool.disconnect(inuse_connections=True)
            if disconnect_from_pool: await disconnect_from_pool
            self._connection = None

    async def keys(self, pattern: str) -> List[str]:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            return await conn.keys(pattern)

    async def mget(self, keys: List[str]) -> List[Any]:
        # This is a heavy operation, definitely needs the semaphore
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            return await conn.mget(keys)
    
    async def publish(self, channel: str, message: Union[str, bytes]) -> int:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            logger.info(f"Publishing message to: {channel}")
            return await conn.publish(channel, message)

    async def _ensure_pubsub(self):
        """Ensure a single PubSub connection and background task are running."""
        # Note: We do NOT use the semaphore here because this creates a long-lived
        # connection that stays open. Using the semaphore would permanently block 1 slot.
        if self._pubsub is None:
            conn = await self._get_or_create_connection()
            self._pubsub = conn.pubsub()
        if self._pubsub_task is None or self._pubsub_task.done():
            self._pubsub_task = asyncio.create_task(self._pubsub_listener(), name="background_redis_pubsub_listener")

    async def _pubsub_listener(self):
        """Single coroutine listening for all messages and dispatching them."""
        # This runs in background and passively receives data.
        # It does NOT use the semaphore to avoid blocking the listener loop.
        try:
            async for message in self._pubsub.listen():  # type: ignore
                if message["type"] != "message":
                    continue

                channel_name = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]

                async with self._sub_lock:
                    if channel_name not in self._channel_subscriptions:
                        continue
                    subscriptions_snapshot = list(self._channel_subscriptions[channel_name].items())

                for subscription_id, sub_info in subscriptions_snapshot:
                    try:
                        message_data = message.copy()
                        if sub_info.decode_responses and isinstance(message_data["data"], bytes):
                            message_data["data"] = message_data["data"].decode("utf-8")

                        # Callbacks run outside the Redis IO semaphore logic 
                        # to prevent application logic from freezing the DB layer
                        if asyncio.iscoroutinefunction(sub_info.callback):
                            await sub_info.callback(message_data, subscription_id)
                        else:
                            sub_info.callback(message_data, subscription_id)

                    except Exception as e:
                        logger.error(f"Error in callback {subscription_id} for channel {channel_name}: {e}")
                        traceback.print_exc()
        except asyncio.CancelledError:
            logger.info("PubSub listener task cancelled")
        except Exception as e:
            logger.error(f"Error in PubSub listener: {e}")
            raise

    async def subscribe(self, channel: str, callback: Callable[[dict, str], Any], decode_responses: bool = False, coroutine_tag: str = "", debug_worker_id: str = "") -> str:
        # Subscribe is a quick control command, we can semaphore it safely
        async with self._op_semaphore:
            await self._simulate_network_latency()
            await self._ensure_pubsub()

            subscription_id = str(uuid.uuid4())
            sub_info = SubscriptionInfo(subscription_id, callback, decode_responses, debug_worker_id)

            async with self._sub_lock:
                first_for_channel = channel not in self._channel_subscriptions
                self._channel_subscriptions.setdefault(channel, {})[subscription_id] = sub_info

            if first_for_channel:
                await self._pubsub.subscribe(channel)  # type: ignore
                logger.info(f"Subscribed to Redis channel {channel}")

            logger.info(f"W({debug_worker_id}) Subscribed to channel: {channel} | tag: {coroutine_tag} | id: {subscription_id}")
            return subscription_id

    async def unsubscribe(self, channel: str, subscription_id: Optional[str] = None):
        async with self._op_semaphore:
            await self._simulate_network_latency()
            async with self._sub_lock:
                if channel not in self._channel_subscriptions:
                    return
                if subscription_id:
                    if subscription_id in self._channel_subscriptions[channel]:
                        del self._channel_subscriptions[channel][subscription_id]
                        logger.info(f"Unsubscribed {subscription_id} from channel {channel}")
                else:
                    self._channel_subscriptions[channel].clear()
                    logger.info(f"Unsubscribed all from channel {channel}")

                # cleanup if empty
                if not self._channel_subscriptions[channel]:
                    del self._channel_subscriptions[channel]
                    await self._pubsub.unsubscribe(channel)  # type: ignore
                    logger.info(f"No more subscribers, unsubscribed Redis channel {channel}")
            
    async def _delete_matching_keys(self, conn: Redis, match_pattern: str) -> int:
        # Note: This helper is called by delete(), which already holds the semaphore.
        # So we do NOT add another semaphore here to avoid deadlocks or redundancy.
        
        keys_to_delete = []
        async for key in conn.scan_iter(match=match_pattern, count=1000):
            keys_to_delete.append(key)
            
        if not keys_to_delete:
            return 0
            
        return await conn.delete(*keys_to_delete)

    async def delete(self, key: str, *, pattern: bool = False, prefix: bool = False, suffix: bool = False) -> int:
        async with self._op_semaphore:
            await self._simulate_network_latency()
            conn = await self._get_or_create_connection()
            
            match_flags = sum([bool(pattern), bool(prefix), bool(suffix)])
            if match_flags > 1:
                raise ValueError("Only one of pattern, prefix, or suffix can be True")
            
            if pattern:
                return await self._delete_matching_keys(conn, key)
            elif prefix:
                return await self._delete_matching_keys(conn, f"{key}*")
            elif suffix:
                return await self._delete_matching_keys(conn, f"*{key}")
            else:
                deleted = await conn.delete(key)
                return deleted