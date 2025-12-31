"""
Cache Manager for Call Transcriptions
Handles local storage and duplicate detection
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class TranscriptionCache:
    """Manages cached transcriptions and duplicate detection."""

    def __init__(self, cache_dir: str = ".transcription_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.load_index()

    def load_index(self):
        """Load the cache index."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def save_index(self):
        """Save the cache index."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def generate_hash(self, transcript: str) -> str:
        """Generate a hash from the transcript text."""
        # Normalize: lowercase, remove extra whitespace
        normalized = ' '.join(transcript.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def check_duplicate(self, transcript: str) -> Optional[Dict]:
        """
        Check if this transcript has been analyzed before.
        Returns the cached result if found, None otherwise.
        """
        transcript_hash = self.generate_hash(transcript)

        if transcript_hash in self.index:
            cache_file = self.cache_dir / f"{transcript_hash}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    cached_data['_from_cache'] = True
                    cached_data['_cache_timestamp'] = self.index[transcript_hash]['timestamp']
                    return cached_data

        return None

    def save_transcription(self, transcript: str, full_data: Dict):
        """Save a new transcription to cache."""
        transcript_hash = self.generate_hash(transcript)
        cache_file = self.cache_dir / f"{transcript_hash}.json"

        # Save the full data
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        # Update index
        self.index[transcript_hash] = {
            'timestamp': datetime.now().isoformat(),
            'file': str(cache_file),
            'preview': transcript[:100] + '...' if len(transcript) > 100 else transcript
        }
        self.save_index()

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        return {
            'total_cached': len(self.index),
            'cache_size_mb': sum(
                f.stat().st_size for f in self.cache_dir.glob('*.json')
            ) / (1024 * 1024),
            'oldest_entry': min(
                (entry['timestamp'] for entry in self.index.values()),
                default=None
            ),
            'newest_entry': max(
                (entry['timestamp'] for entry in self.index.values()),
                default=None
            )
        }

    def list_cached_items(self) -> List[Dict]:
        """List all cached transcriptions."""
        items = []
        for hash_key, info in self.index.items():
            items.append({
                'hash': hash_key,
                'timestamp': info['timestamp'],
                'preview': info['preview']
            })
        return sorted(items, key=lambda x: x['timestamp'], reverse=True)

    def clear_cache(self):
        """Clear all cached data."""
        for file in self.cache_dir.glob('*.json'):
            file.unlink()
        self.index = {}
        self.save_index()
