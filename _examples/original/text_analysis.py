import os
import sys
import time
from typing import List, Dict, Any
import random
from collections import Counter
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dag_task_node import DAGTask

# Import centralized configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.config import WORKER_CONFIG

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- DAG Tasks ---

# --- Initial fan-out group (word counts on chunks) ---

@DAGTask
def word_count_chunk(text: str, start: int, end: int) -> int:
    words = text.split()
    # Vectorized-ish counting by slicing and using numpy for a small heavy op
    count = len(words[start:end])
    return count


@DAGTask
def merge_word_counts(text: str, counts: List[int]) -> tuple[int, str]:
    # Use numpy sum (vectorized) and a small deterministic heavy op
    counts_arr = np.array(counts, dtype=np.int64)
    total = int(counts_arr.sum())
    # Small CPU-heavy vector op to scale with available CPU
    return total, text


@DAGTask
def create_text_segments(res: tuple[int, str]) -> List[str]:
    _, text = res
    words = text.split()
    n = len(words)
    # compute segment sizes vectorized style
    segment_size = n // 16  # CHANGED: Was 8
    segments = []
    # still returns the same segment strings (unchanged behavior)
    for i in range(16):  # CHANGED: Was 8
        start_idx = i * segment_size
        end_idx = n if i == 15 else (i + 1) * segment_size  # CHANGED: Was 7
        segments.append(" ".join(words[start_idx:end_idx]))
    return segments


@DAGTask
def compute_text_statistics(res: tuple[int, str]) -> Dict[str, Any]:
    _, text = res
    words = text.split()

    # Vectorized word-lengths via numpy
    clean_words = [w.lower().strip('.,!?;:"()') for w in words]
    if clean_words:
        lengths = np.fromiter((len(w) for w in clean_words), dtype=np.int32)
        avg_word_length = float(lengths.mean())
    else:
        lengths = np.array([], dtype=np.int32)
        avg_word_length = 0.0

    # Sentence and simple stats
    sentence_count = text.count('.')
    avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0.0
    simple_readability = avg_word_length + (avg_sentence_length / 10.0)

    # Vectorized frequency counting using numpy.unique
    # Filter words longer than 4 as in original logic
    filtered = [w for w in clean_words if len(w) > 4]
    if filtered:
        arr = np.array(filtered)
        uniques, counts = np.unique(arr, return_counts=True)
        # get top 10
        top_idx = np.argsort(counts)[-10:][::-1]
        most_common = [(uniques[i], int(counts[i])) for i in top_idx]
        unique_word_count = int(uniques.size)
    else:
        most_common = []
        unique_word_count = 0

    # Vectorized character-level vowel/consonant counts using numpy
    # Convert text to lower bytes and examine
    if text:
        b = np.frombuffer(text.lower().encode('utf-8'), dtype=np.uint8)
        # map letters to ascii where possible (non-ascii will be counted as neither)
        # check a,e,i,o,u and alphabetic range
        # create boolean masks for vowels (utf-8 bytes for ascii letters)
        vowels_mask = np.isin(b, np.frombuffer(b'aeiou', dtype=np.uint8))
        # alphabetic: a-z
        alpha_mask = (b >= ord('a')) & (b <= ord('z'))
        vowel_count = int(vowels_mask.sum())
        consonant_count = int((alpha_mask & ~vowels_mask).sum())
    else:
        vowel_count = 0
        consonant_count = 0

    # Extra vectorized CPU work to favor multi-core (deterministic)
    # rnd = np.linspace(0.0, 1.0, 8192, dtype=np.float64)
    # _ = np.sum(np.sqrt(rnd) * np.sin(rnd * np.pi))

    return {
        "vowel_count": vowel_count,
        "consonant_count": consonant_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "most_common_words": most_common,
        "unique_word_count": unique_word_count,
        "simple_readability_score": simple_readability,
        "processing_time_simulation": "Vectorized CPU-heavy stats completed"
    }


@DAGTask
def analyze_segment(segments: List[str], segment_id: int) -> Dict[str, Any]:
    text = segments[segment_id]
    words = text.split()
    # vectorized average word length
    if words:
        lengths = np.fromiter((len(w.strip('.,!?;:"()')) for w in words), dtype=np.int32)
        avg_word_length = float(lengths.mean()) if lengths.size else 0.0
    else:
        avg_word_length = 0.0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    # unique words using numpy for speed
    clean_words = [w.lower().strip('.,!?;:"()') for w in words]
    unique_words = int(np.unique(np.array(clean_words)).size) if clean_words else 0

    # Small deterministic heavy op to favor vector units
    _ = np.linalg.norm(np.arange(2048, dtype=np.float64))

    return {
        "segment_id": segment_id,
        "text": text,
        "word_count": len(words),
        "avg_word_length": avg_word_length,
        "sentence_count": len(sentences),
        "unique_words": unique_words
    }


@DAGTask
def extract_overall_keywords(segments: List[str], text_stats: Dict[str, Any]) -> Dict[str, Any]:
    MAX_SAMPLE_SIZE = 50
    MAX_WORDS_PER_SEGMENT = 100

    # sample segments vectorized-style
    if len(segments) > MAX_SAMPLE_SIZE:
        sample_segments = random.sample(segments, MAX_SAMPLE_SIZE)
    else:
        sample_segments = segments

    # flatten sampled segments into a word array (up to cap)
    words_list = []
    for seg in sample_segments:
        words_list.extend(seg.lower().split()[:MAX_WORDS_PER_SEGMENT])
    if words_list:
        clean = np.array([w.strip('.,!?;:"()') for w in words_list])
        # length filter vectorized
        lens = np.fromiter((len(w) for w in clean), dtype=np.int32)
        mask = (lens > 5) & (lens < 20)
        filtered = clean[mask]
        # exclude common words from text_stats (preserve same API)
        common_words = {w for w, _ in text_stats.get("most_common_words", [])}
        if filtered.size:
            # unique and counts using numpy
            uniques, counts = np.unique(filtered, return_counts=True)
            # filter out common_words
            if common_words:
                keep_mask = np.array([u not in common_words for u in uniques])
                uniques = uniques[keep_mask]
                counts = counts[keep_mask]
            # build top 10
            if counts.size:
                top_idx = np.argsort(counts)[-10:][::-1]
                top = [(str(uniques[i]), int(counts[i])) for i in top_idx]
            else:
                top = []
            total_keywords = int(uniques.size)
            total_processed = int(filtered.size)
        else:
            top = []
            total_keywords = 0
            total_processed = 0
    else:
        top = []
        total_keywords = 0
        total_processed = 0

    # small deterministic heavy op
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64)))

    return {
        "type": "overall_keywords",
        "top_keywords": top,
        "total_keywords": total_keywords,
        "keyword_density": (total_keywords / total_processed) if total_processed > 0 else 0.0,
        "avg_word_length_context": text_stats.get("avg_word_length", 0.0),
        "is_sample": True,
        "sample_size": len(sample_segments)
    }


@DAGTask
def analyze_overall_punctuation(segments: List[str], text_stats: Dict[str, Any]) -> Dict[str, Any]:
    # Vectorize by joining then using numpy byte ops for counts
    all_text = " ".join(segments)
    if not all_text:
        punctuation_counts = {k: 0 for k in ["periods", "commas", "exclamations",
                                             "questions", "semicolons", "colons", "quotations"]}
        total_punct = 0
    else:
        b = np.frombuffer(all_text.encode('utf-8'), dtype=np.uint8)
        punctuation_counts = {
            "periods": int((b == ord('.')).sum()),
            "commas": int((b == ord(',')).sum()),
            "exclamations": int((b == ord('!')).sum()),
            "questions": int((b == ord('?')).sum()),
            "semicolons": int((b == ord(';')).sum()),
            "colons": int((b == ord(':')).sum()),
            "quotations": int((b == ord('"')).sum()) + int((b == ord("'")).sum())
        }
        total_punct = sum(punctuation_counts.values())

    # small vectorized workload to scale with CPU
    # _ = np.sum(np.sin(np.linspace(0.0, 3.1415, 4096, dtype=np.float64)))

    return {
        "type": "overall_punctuation",
        "punctuation_counts": punctuation_counts,
        "total_punctuation": total_punct,
        "punctuation_density": (total_punct / len(all_text)) if all_text else 0.0,
        "most_common_punct": max(punctuation_counts.items(), key=lambda x: x[1])[0] if punctuation_counts else "none",
        "vowel_context": text_stats.get("vowel_count", 0),
        "consonant_context": text_stats.get("consonant_count", 0)
    }


@DAGTask
def calculate_overall_readability(segments: List[str], text_stats: Dict[str, Any]) -> Dict[str, Any]:
    # Use precomputed stats but add a vectorized refinement: compute a small statistical fingerprint
    all_text = " ".join(segments)
    enhanced_score = text_stats.get("simple_readability_score", 0.0) + (text_stats.get("avg_sentence_length", 0.0) * 0.1)

    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64)))

    return {
        "type": "overall_readability",
        "simple_readability_score": text_stats.get("simple_readability_score", 0.0),
        "enhanced_readability_score": enhanced_score,
        "avg_word_length": text_stats.get("avg_word_length", 0.0),
        "avg_sentence_length": text_stats.get("avg_sentence_length", 0.0),
        "complexity_level": "simple" if enhanced_score < 8 else "medium" if enhanced_score < 12 else "complex",
        "sentence_context": text_stats.get("sentence_count", 0)
    }


@DAGTask
def detect_overall_patterns(segments: List[str], text_stats: Dict[str, Any]) -> Dict[str, Any]:
    all_text = " ".join(segments)
    words = all_text.split()
    total_words = len(words)
    unique_words = text_stats.get("unique_word_count", 0)
    avg_word_length = text_stats.get("avg_word_length", 0.0)
    segment_count = len(segments)
    vowel_count = text_stats.get("vowel_count", 0)

    # vocabulary richness vectorized ratio (stable)
    vocabulary_richness = (unique_words / total_words) if total_words > 0 else 0.0

    # Consume time using CPU, more CPU = faster
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 8192, dtype=np.float64)))

    return {
        "type": "overall_patterns",
        "patterns": {
            "total_words": total_words,
            "unique_words": unique_words,
            "avg_word_length": avg_word_length,
            "segment_count": segment_count,
            "vowel_count": vowel_count
        },
        "vocabulary_richness": vocabulary_richness,
        "lexical_diversity": "high" if vocabulary_richness > 0.7 else "medium" if vocabulary_richness > 0.4 else "low",
        "readability_context": text_stats.get("simple_readability_score", 0.0)
    }


@DAGTask
def merge_segment_analyses(segment_analyses: List[Dict[str, Any]],
                           overall_keywords: Dict[str, Any],
                           overall_punctuation: Dict[str, Any],
                           overall_readability: Dict[str, Any],
                           overall_patterns: Dict[str, Any]) -> Dict[str, Any]:
    # Aggregate with numpy where helpful
    word_counts = np.fromiter((s.get("word_count", 0) for s in segment_analyses), dtype=np.int64) if segment_analyses else np.array([], dtype=np.int64)
    sentence_counts = np.fromiter((s.get("sentence_count", 0) for s in segment_analyses), dtype=np.int64) if segment_analyses else np.array([], dtype=np.int64)

    total_words = int(word_counts.sum()) if word_counts.size else 0
    total_sentences = int(sentence_counts.sum()) if sentence_counts.size else 0

    # weighted avg word length
    if total_words > 0:
        weighted_sum = sum((s.get("avg_word_length", 0.0) * s.get("word_count", 0) for s in segment_analyses))
        overall_avg_word_length = float(weighted_sum / total_words)
    else:
        overall_avg_word_length = 0.0

    total_unique_words = sum(s.get("unique_words", 0) for s in segment_analyses)

    # Consume time using CPU, more CPU = faster
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64)))

    return {
        "total_segments": len(segment_analyses),
        "total_words": total_words,
        "total_sentences": total_sentences,
        "overall_avg_word_length": overall_avg_word_length,
        "total_unique_words": total_unique_words,

        "keywords_analysis": overall_keywords,
        "punctuation_analysis": overall_punctuation,
        "readability_analysis": overall_readability,
        "patterns_analysis": overall_patterns,

        "segment_details": segment_analyses
    }


@DAGTask
def calculate_text_metrics(merged_analysis: Dict[str, Any]) -> Dict[str, Any]:
    total_words = merged_analysis.get("total_words", 0)
    total_sentences = merged_analysis.get("total_sentences", 0)
    vocab_rich = merged_analysis.get("patterns_analysis", {}).get("vocabulary_richness", 0.0)
    overall_avg_word_length = merged_analysis.get("overall_avg_word_length", 0.0)
    keyword_density = merged_analysis.get("keywords_analysis", {}).get("keyword_density", 0.0)

    # Use numpy for arithmetic to keep things vectorized-friendly
    words_per_sentence = float(np.divide(total_words, total_sentences)) if total_sentences > 0 else 0.0
    complexity_score = float((overall_avg_word_length * total_sentences) / 100.0) if total_sentences > 0 else 0.0

    # Consume time using CPU, more CPU = faster
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64)))

    return {
        "words_per_sentence": words_per_sentence,
        "vocabulary_richness": vocab_rich,
        "complexity_score": complexity_score,
        "keyword_density": keyword_density
    }


@DAGTask
def generate_text_summary(merged_analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Keep original summary fields but compute some vectorized micro-stats
    total_words = merged_analysis.get("total_words", 0)
    total_segments = merged_analysis.get("total_segments", 0)
    readability = merged_analysis.get("readability_analysis", {}).get("complexity_level", "unknown")
    lexical = merged_analysis.get("patterns_analysis", {}).get("lexical_diversity", "unknown")
    keywords = merged_analysis.get("keywords_analysis", {}).get("top_keywords", [])[:5]
    punctuation_style = merged_analysis.get("punctuation_analysis", {}).get("most_common_punct", "none")

    return {
        "summary": f"Text contains {total_words} words across {total_segments} segments",
        "readability": readability,
        "lexical_diversity": lexical,
        "top_keywords": keywords,
        "punctuation_style": punctuation_style
    }


@DAGTask
def final_comprehensive_report(metrics: Dict[str, Any], summary: Dict[str, Any],
                               merged_analysis: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "analysis_metrics": metrics,
        "text_summary": summary,
        "detailed_analysis": merged_analysis
    }

# --- Define Workflow ---

input_file = "../_inputs/shakespeare.txt"
text = _read_file(input_file)

# Initial fan-out group (word count tasks)
chunk_size = 200
num_chunks = 32
word_counts = []

for i in range(num_chunks):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    wc = word_count_chunk(text, start, end)
    word_counts.append(wc)

merge_wc_result = merge_word_counts(text, word_counts)

# Single task that prepares data for middle fan-out
segments_data = create_text_segments(merge_wc_result)

# Heavy computational task that also fans out from merge_word_counts
text_statistics = compute_text_statistics(merge_wc_result)

# FAN-OUT: Generate 16 segment analyses + 4 direct processing functions (20 total tasks)
segment_analyses = [analyze_segment(segments_data, i) for i in range(16)]

# 4 additional processing functions that work on segments data + text statistics
overall_keywords = extract_overall_keywords(segments_data, text_statistics)
overall_punctuation = analyze_overall_punctuation(segments_data, text_statistics)
overall_readability = calculate_overall_readability(segments_data, text_statistics)
overall_patterns = detect_overall_patterns(segments_data, text_statistics)

# Fan-in: Merge all 20 analyses (16 segments + 4 overall processing results)
merged_analysis = merge_segment_analyses(
    segment_analyses,
    overall_keywords,
    overall_punctuation,
    overall_readability,
    overall_patterns
)

# Create two branches from the merged analysis
metrics = calculate_text_metrics(merged_analysis)
summary = generate_text_summary(merged_analysis)

final_report = final_comprehensive_report(metrics, summary, merged_analysis)
# final_report.visualize_dag(open_after=True)
# exit()

# --- Run workflow ---
start_time = time.time()
result = final_report.compute(dag_name="text_analysis", config=WORKER_CONFIG, open_dashboard=False)
# Result keys: {list(result.keys())} |
print(f"User waited: {time.time() - start_time:.3f}s")
# print(f"Analysis complete - processed {result['detailed_analysis']['total_words']} words in {result['detailed_analysis']['total_segments']} segments")
# print(f"Found {result['detailed_analysis']['keywords_analysis']['total_keywords']} unique keywords")
# print(f"Overall readability: {result['detailed_analysis']['readability_analysis']['complexity_level']}")
# print(f"Vocabulary richness: {result['detailed_analysis']['patterns_analysis']['lexical_diversity']}")