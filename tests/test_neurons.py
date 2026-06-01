"""
Unit Tests for Neuron Architectures
==================================
"""

import numpy as np
import pytest
from neuron_arch import (
    AdaptiveThresholdSCH,
    ResidualPCLayer,
    PatternSeparationHM,
    NormalizedNG,
    ESBNeuronLayer,
    IntegratedSystem
)


class TestSCHNeuron:
    """Tests for SCH-Neuron."""
    
    def test_initialization(self):
        """Test proper initialization."""
        sch = AdaptiveThresholdSCH(n_neurons=64)
        assert sch.n == 64
        assert len(sch.v) == 64
        assert len(sch.z) == 64
        assert np.all(sch.v_th == sch.v_th_base)
    
    def test_spike_generation(self):
        """Test that high input generates spikes."""
        sch = AdaptiveThresholdSCH(n_neurons=64, v_th_base=0.1)
        high_input = np.ones(64) * 2.0
        
        spikes, continuous = sch.step(high_input)
        
        assert np.sum(spikes) > 0, "High input should generate spikes"
    
    def test_adaptive_threshold(self):
        """Test that threshold adapts to maintain target rate."""
        sch = AdaptiveThresholdSCH(
            n_neurons=64, 
            target_spike_rate=0.1,
            adapt_strength=1.0
        )
        
        for _ in range(100):
            input_current = np.random.randn(64) * 0.5
            sch.step(input_current)
        
        avg_rate = np.mean(sch.spike_history)
        assert 0.01 < avg_rate < 0.3, f"Rate {avg_rate} should be near target 0.1"
    
    def test_sparsity_computation(self):
        """Test sparsity calculation."""
        sch = AdaptiveThresholdSCH(n_neurons=64)
        
        for _ in range(10):
            sch.step(np.random.randn(64) * 0.3)
        
        sparsity = sch.get_sparsity()
        assert 0.0 <= sparsity <= 1.0
    
    def test_reset(self):
        """Test reset functionality."""
        sch = AdaptiveThresholdSCH(n_neurons=64)
        
        for _ in range(10):
            sch.step(np.random.randn(64))
        
        sch.reset()
        
        assert np.all(sch.v == 0)
        assert np.all(sch.z == 0)
        assert len(sch.spike_history) == 0


class TestPCNeuron:
    """Tests for PC-Neuron."""
    
    def test_initialization(self):
        """Test proper initialization."""
        pc = ResidualPCLayer(n_state=64, n_pred=64)
        assert pc.W.shape == (64, 64)
    
    def test_prediction(self):
        """Test prediction generation."""
        pc = ResidualPCLayer(n_state=64, n_pred=64)
        state = np.random.randn(64)
        
        prediction = pc.predict(state)
        
        assert len(prediction) == 64
        assert not np.any(np.isnan(prediction))
    
    def test_learning(self):
        """Test local learning rule."""
        pc = ResidualPCLayer(n_state=64, n_pred=64, lr=0.01)
        
        W_before = pc.W.copy()
        
        for _ in range(10):
            state = np.random.randn(64)
            target = np.random.randn(64)
            pc.predict(state)
            pc.local_learn(state, target)
        
        assert not np.allclose(pc.W, W_before), "Weights should change"
    
    def test_gradient_clipping(self):
        """Test gradient clipping prevents explosion."""
        pc = ResidualPCLayer(n_state=64, n_pred=64, lr=0.01)
        
        for _ in range(100):
            state = np.random.randn(64) * 10
            target = np.random.randn(64) * 10
            pc.predict(state)
            pc.local_learn(state, target)
        
        assert np.linalg.norm(pc.W) < 100, "Weights should remain bounded"


class TestHMNeuron:
    """Tests for HM-Neuron."""
    
    def test_initialization(self):
        """Test proper initialization."""
        hm = PatternSeparationHM(n_input=64, n_hidden=32)
        assert hm.W_hippo.shape == (32, 64)
        assert hm.W_cortex.shape == (32, 64)
    
    def test_pattern_separation(self):
        """Test pattern separation creates sparse output."""
        hm = PatternSeparationHM(n_input=64, n_hidden=32, top_k=8)
        x = np.random.randn(64)
        
        output = hm.pattern_separation(x)
        
        non_zero = np.sum(output != 0)
        assert non_zero <= 8, "Should have at most top_k non-zero elements"
    
    def test_memory_storage(self):
        """Test that memories are stored."""
        hm = PatternSeparationHM(n_input=64, n_hidden=32)
        
        for i in range(10):
            x = np.random.randn(64)
            target = np.random.randn(32)
            hm.learn(x, target)
        
        assert len(hm.memory_buffer) == 10
    
    def test_consolidation(self):
        """Test consolidation uses replay."""
        hm = PatternSeparationHM(n_input=64, n_hidden=32)
        
        for i in range(20):
            x = np.random.randn(64)
            target = np.random.randn(32)
            hm.learn(x, target)
        
        n_replay = hm.consolidate(n_replay=10)
        
        assert n_replay == 10
        assert hm.consolidation_count == 1


class TestNGNeuron:
    """Tests for NG-Neuron."""
    
    def test_initialization(self):
        """Test proper initialization."""
        ng = NormalizedNG()
        assert 0 <= ng.da <= 1
        assert 0 <= ng.serotonin <= 1
    
    def test_gate_computation(self):
        """Test gate is in valid range."""
        ng = NormalizedNG()
        
        gate = ng.compute_gate()
        
        assert 0.1 <= gate <= 0.9
    
    def test_effective_lr_range(self):
        """Test effective learning rate is bounded."""
        ng = NormalizedNG(base_lr=0.01)
        
        for _ in range(100):
            ng.update_from_reward(np.random.rand())
            lr = ng.compute_effective_lr()
            assert 0.002 <= lr <= 0.02
    
    def test_reward_updates_da(self):
        """Test that reward updates dopamine."""
        ng = NormalizedNG()
        
        da_before = ng.da
        ng.update_from_reward(1.0)
        da_after_high = ng.da
        
        ng.reset()
        ng.update_from_reward(0.0)
        da_after_low = ng.da
        
        assert da_after_high > da_after_low


class TestESBNeuron:
    """Tests for ESB-Neuron."""
    
    def test_initialization(self):
        """Test proper initialization."""
        esb = ESBNeuronLayer(n_embodied=16, n_latent=8, n_symbols=4)
        assert esb.W_encode.shape == (8, 16)
        assert esb.W_decode.shape == (4, 8)
    
    def test_encoding(self):
        """Test embodied encoding."""
        esb = ESBNeuronLayer(n_embodied=16, n_latent=8, n_symbols=4)
        sensor = np.random.randn(16)
        
        latent = esb.encode_embodied(sensor)
        
        assert len(latent) == 8
        assert np.all(np.abs(latent) <= 1)  # tanh output
    
    def test_symbol_decoding(self):
        """Test symbol decoding."""
        esb = ESBNeuronLayer(n_embodied=16, n_latent=8, n_symbols=4)
        latent = np.random.randn(8)
        
        symbol_id, probs, confidence = esb.decode_symbol(latent)
        
        assert 0 <= symbol_id < 4
        assert len(probs) == 4
        assert abs(np.sum(probs) - 1.0) < 1e-6
        assert 0 <= confidence <= 1
    
    def test_grounding(self):
        """Test symbol grounding."""
        esb = ESBNeuronLayer(n_embodied=16, n_latent=8, n_symbols=4)
        sensor = np.random.randn(16)
        
        esb.ground_symbol(0, sensor, strength=0.5)
        
        grounded = esb.retrieve_embodied(0)
        assert len(grounded) == 16


class TestIntegratedSystem:
    """Tests for integrated system."""
    
    def test_initialization(self):
        """Test proper initialization."""
        system = IntegratedSystem(n_sch=64, n_pc=64, n_hm=32)
        
        params = system.count_parameters()
        assert params['total'] > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        system = IntegratedSystem(n_sch=64, n_pc=64, n_hm=32)
        
        result = system.forward(0.5)
        
        assert 'output' in result
        assert 'spikes' in result
        assert isinstance(result['output'], float)
    
    def test_learning(self):
        """Test learning step."""
        system = IntegratedSystem(n_sch=64, n_pc=64, n_hm=32)
        
        metrics = system.learn(0.5, 0.7)
        
        assert 0 <= metrics.sparsity <= 1
        assert metrics.grad_norm >= 0
        assert 0 <= metrics.da <= 1
    
    def test_consolidation(self):
        """Test system consolidation."""
        system = IntegratedSystem(n_sch=64, n_pc=64, n_hm=32)
        
        for _ in range(10):
            system.learn(np.random.rand(), np.random.rand())
        
        n_replay = system.consolidate()
        
        assert n_replay > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
