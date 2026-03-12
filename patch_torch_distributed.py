"""
Patch for torch.distributed to support ROCm builds that don't include ReduceOp.

This patch should be imported BEFORE any libraries that use torch.distributed
(like encodec, vocos, etc.)

Usage:
    import patch_torch_distributed  # Must be first!
    from zipvoice.luxvoice import LuxTTS  # Now safe to import
"""

import torch

# Check if torch.distributed needs patching
if not hasattr(torch.distributed, 'ReduceOp'):
    print("⚠️  Applying torch.distributed patch for ROCm compatibility...")
    
    # Create a mock ReduceOp class
    class _MockReduceOp:
        """Mock ReduceOp for PyTorch builds without distributed support (e.g., ROCm)"""
        SUM = "sum"
        PRODUCT = "product" 
        MIN = "min"
        MAX = "max"
        BAND = "band"
        BOR = "bor"
        BXOR = "bxor"
    
    # Inject the mock into torch.distributed
    torch.distributed.ReduceOp = _MockReduceOp()
    
    print("✓ torch.distributed.ReduceOp patched successfully")
    print("  Note: Distributed training is not available, but inference will work")
else:
    print("✓ torch.distributed.ReduceOp already available (no patch needed)")
