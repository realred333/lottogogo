"""Hidden Markov Model based scorer for lottery numbers.

This module uses HMM to detect Hot/Neutral/Cold states for each number
based on their appearance patterns in historical data.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn import hmm

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_NUMBERS = 45

# State definitions
STATE_HOT = 0
STATE_NEUTRAL = 1
STATE_COLD = 2
STATE_NAMES = {STATE_HOT: "hot", STATE_NEUTRAL: "neutral", STATE_COLD: "cold"}


class HMMScorer:
    """Calculate boosts for numbers based on HMM state inference.
    
    Uses a 3-state Hidden Markov Model to detect:
    - Hot state: Number is in a "hot streak", appearing frequently
    - Neutral state: Normal appearance pattern
    - Cold state: Number is in a "cold streak", appearing rarely
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        hot_boost: float = 0.3,
        cold_boost: float = 0.15,  # Cold = opportunity
        window: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize HMM scorer.
        
        Args:
            n_states: Number of hidden states (default 3: Hot/Neutral/Cold)
            n_iter: Number of EM iterations for training
            hot_boost: Boost value for numbers in Hot state
            cold_boost: Boost value for numbers in Cold state (opportunity)
            window: Number of recent rounds to use for training
            random_state: Random seed for reproducibility
        """
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if window < 20:
            raise ValueError("window must be >= 20 for meaningful training")
        
        self.n_states = n_states
        self.n_iter = n_iter
        self.hot_boost = float(hot_boost)
        self.cold_boost = float(cold_boost)
        self.window = window
        self.random_state = random_state
        
        # Store trained models for each number
        self._models: dict[int, hmm.CategoricalHMM] = {}
        self._states: dict[int, int] = {}

    def _create_binary_sequence(
        self, history: pd.DataFrame, number: int
    ) -> np.ndarray:
        """Create binary sequence for a number (1=appeared, 0=not appeared)."""
        sequence = []
        for _, row in history.iterrows():
            appeared = any(int(row[col]) == number for col in NUMBER_COLUMNS)
            sequence.append(1 if appeared else 0)
        return np.array(sequence).reshape(-1, 1)

    def _fit_single_number(
        self, history: pd.DataFrame, number: int
    ) -> tuple[hmm.CategoricalHMM | None, int]:
        """Train HMM for a single number and return predicted current state."""
        sequence = self._create_binary_sequence(history, number)
        
        if len(sequence) < 20:
            return None, STATE_NEUTRAL
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            
            try:
                model.fit(sequence)
                
                # Predict hidden states for the sequence
                hidden_states = model.predict(sequence)
                
                # Get the current state (last state)
                current_state = int(hidden_states[-1])
                
                # Classify state based on emission probabilities
                # Higher emission prob for "1" (appeared) = Hot state
                emission_probs = model.emissionprob_
                
                # Sort states by their probability of emitting "1" (appeared)
                state_hotness = [(s, emission_probs[s, 1] if emission_probs.shape[1] > 1 else 0.5) 
                                 for s in range(self.n_states)]
                state_hotness.sort(key=lambda x: x[1], reverse=True)
                
                # Map current state to Hot/Neutral/Cold based on emission ranking
                state_mapping = {
                    state_hotness[0][0]: STATE_HOT,     # Highest emission = Hot
                    state_hotness[-1][0]: STATE_COLD,   # Lowest emission = Cold
                }
                for i in range(1, len(state_hotness) - 1):
                    state_mapping[state_hotness[i][0]] = STATE_NEUTRAL
                
                mapped_state = state_mapping.get(current_state, STATE_NEUTRAL)
                
                return model, mapped_state
                
            except Exception:
                return None, STATE_NEUTRAL

    def fit(self, history: pd.DataFrame) -> None:
        """Train HMM models for all numbers."""
        if history.empty:
            raise ValueError("history cannot be empty")
        
        # Use only recent window
        if len(history) > self.window:
            if "round" in history.columns:
                history = history.sort_values("round").tail(self.window)
            else:
                history = history.tail(self.window)
        
        self._models.clear()
        self._states.clear()
        
        for number in range(1, TOTAL_NUMBERS + 1):
            model, state = self._fit_single_number(history, number)
            if model is not None:
                self._models[number] = model
            self._states[number] = state

    def get_states(self) -> dict[int, str]:
        """Return current state names for all numbers."""
        return {n: STATE_NAMES.get(s, "neutral") for n, s in self._states.items()}

    def calculate_boosts(
        self, history: pd.DataFrame
    ) -> tuple[dict[int, float], dict[int, list[str]]]:
        """Calculate boost scores based on HMM states.
        
        Returns:
            Tuple of (boosts dict, tags dict) similar to BoostCalculator
        """
        self.fit(history)
        
        boosts: dict[int, float] = {}
        tags: dict[int, list[str]] = {}
        
        for number in range(1, TOTAL_NUMBERS + 1):
            state = self._states.get(number, STATE_NEUTRAL)
            
            if state == STATE_HOT:
                boosts[number] = self.hot_boost
                tags[number] = ["hmm_hot"]
            elif state == STATE_COLD:
                boosts[number] = self.cold_boost
                tags[number] = ["hmm_cold"]
            else:
                boosts[number] = 0.0
                tags[number] = []
        
        return boosts, tags

    def get_summary(self) -> dict[str, list[int]]:
        """Get summary of numbers in each state."""
        hot_numbers = [n for n, s in self._states.items() if s == STATE_HOT]
        cold_numbers = [n for n, s in self._states.items() if s == STATE_COLD]
        neutral_numbers = [n for n, s in self._states.items() if s == STATE_NEUTRAL]
        
        return {
            "hot": sorted(hot_numbers),
            "cold": sorted(cold_numbers),
            "neutral": sorted(neutral_numbers),
        }
