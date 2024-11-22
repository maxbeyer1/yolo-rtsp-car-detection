"""
Module to collect system metrics in the background.
"""
import time
import threading
import logging
from datetime import datetime
import json
from typing import Dict, List
from pathlib import Path
import numpy as np
import psutil


class SystemMetricsCollector:
    """
    Class to collect system metrics in the background
    """

    def __init__(self, output_dir: Path, sample_interval: float = 1.0):
        self.output_dir = output_dir / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval = sample_interval
        self.is_running = False
        self.metrics_buffer: List[Dict] = []
        self.logger = logging.getLogger(__name__)

        # Track processing times
        self.processing_times: List[float] = []

    def start(self):
        """Start collecting system metrics"""
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.start()

    def stop(self):
        """Stop collecting system metrics"""
        self.is_running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join()
        self._save_summary()

    def record_processing_time(self, duration: float):
        """Record the processing time for a single frame"""
        self.processing_times.append(duration)

    def _collect_metrics(self):
        """Collect system metrics in the background"""
        while self.is_running:
            try:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu': {
                        'total_percent': psutil.cpu_percent(interval=None),
                        'per_cpu_percent': psutil.cpu_percent(interval=None, percpu=True),
                        'load_avg': psutil.getloadavg(),
                    },
                    'memory': {
                        'total': psutil.virtual_memory().total,
                        'available': psutil.virtual_memory().available,
                        'percent': psutil.virtual_memory().percent,
                        'used': psutil.virtual_memory().used,
                    },
                    'disk': {
                        'read_bytes': psutil.disk_io_counters().read_bytes,
                        'write_bytes': psutil.disk_io_counters().write_bytes,
                    }
                }

                self.metrics_buffer.append(metrics)

                # Save buffer every 60 samples
                if len(self.metrics_buffer) >= 60:
                    self._save_metrics()

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")

            time.sleep(self.sample_interval)

    def _save_metrics(self):
        """Save collected metrics to a JSON file"""
        if not self.metrics_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"metrics_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(self.metrics_buffer, f, indent=2)

        self.metrics_buffer = []

    def _save_summary(self):
        """Save statistical summary of the collected metrics"""
        if not self.processing_times:
            return

        processing_times_array = np.array(self.processing_times)

        summary = {
            'frame_processing_stats': {
                'mean_time': float(np.mean(processing_times_array)),
                'median_time': float(np.median(processing_times_array)),
                'std_time': float(np.std(processing_times_array)),
                'min_time': float(np.min(processing_times_array)),
                'max_time': float(np.max(processing_times_array)),
                'p95_time': float(np.percentile(processing_times_array, 95)),
                'total_frames_processed': len(self.processing_times)
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"summary_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
