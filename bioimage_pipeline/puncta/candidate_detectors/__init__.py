"""Pluggable image-level candidate detectors."""

from bioimage_pipeline.puncta.candidate_detectors.base import CandidateDetector, get_detector
from bioimage_pipeline.puncta.candidate_detectors.python_log import PythonLoGDetector

__all__ = ["CandidateDetector", "PythonLoGDetector", "get_detector"]
