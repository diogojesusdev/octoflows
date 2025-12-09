from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Union, Optional

class Storage(ABC):
    @dataclass
    class Config(ABC):
        @abstractmethod
        def create_instance(self) -> "Storage": pass

    @abstractmethod
    async def get(self, key: str) -> Any: pass
    
    @abstractmethod
    async def mget(self, keys: list[str]) -> list[Any]: pass
    
    @abstractmethod
    async def exists(self, *keys: str) -> Any: pass # returns number of keys that exist
    
    @abstractmethod
    async def set(self, key: str, value) -> Any: pass
    
    @abstractmethod
    async def atomic_increment_and_get(self, key: str) -> Any: pass
    
    @abstractmethod
    async def keys(self, pattern: str) -> list: pass

    # PUB/SUB
    @abstractmethod
    async def publish(self, channel: str, message: Union[str, bytes]) -> int: pass
    
    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[dict, str], Any], decode_responses: bool = False, coroutine_tag: str = "", debug_worker_id: str = "") -> str: pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str, subscription_id: Optional[str]): pass

    @abstractmethod
    async def delete(self, key: str, *, pattern: bool = False, prefix: bool = False, suffix: bool = False) -> int:
        """
        Delete keys from the storage.
        
        Args:
            key: The key pattern/prefix/suffix to match for deletion
            pattern: If True, treat key as a pattern to match (e.g., 'user:*')
            prefix: If True, delete all keys that start with the given key
            suffix: If True, delete all keys that end with the given key
            
        Returns:
            int: Number of keys that were deleted
            
        Note:
            Only one of pattern, prefix, or suffix should be True at a time.
            If none are True, performs an exact key match delete.
        """
        pass
        
    @abstractmethod
    async def close_connection(self): pass