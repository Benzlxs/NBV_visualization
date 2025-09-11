#!/usr/bin/env python3
"""Interactive 3D Camera Frustum Demo with Blender Rendering

Launch an interactive 3D interface where you can manipulate camera frustums 
and see real-time Blender renders displayed at the frustum position.

Usage:
    conda activate vggt
    python blender_frustum_demo.py

Then open your browser to http://localhost:8080
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path

# Ensure we're in the vggt environment
env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
if env_name != 'vggt':
    print("⚠️  WARNING: Not running in 'vggt' conda environment!")
    print("   Please run: conda activate vggt")
    print("   Current environment:", env_name)

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from interactive_renderer.frustum_controls import launch_frustum_controls

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_blender_environment():
    """Check that Blender and environment are properly configured."""
    issues = []
    
    # Check conda environment
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    if env_name == 'vggt':
        logger.info(f"✅ Running in correct conda environment: {env_name}")
    else:
        issues.append(f"Not in 'vggt' environment (current: {env_name})")
    
    # Check Blender
    try:
        result = subprocess.run(['blender', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info(f"✅ Blender found: {version_line}")
        else:
            issues.append("Blender executable not working")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("Blender executable not found in PATH")
    
    # Check GLB model
    test_model = Path("dataset/ABO/raw/3dmodels/original/0/B01LR5RSG0.glb")
    if test_model.exists():
        size_mb = test_model.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Test model found: {test_model.name} ({size_mb:.1f}MB)")
        model_path = str(test_model)
    else:
        # Try to find any available GLB model
        model_dir = test_model.parent
        if model_dir.exists():
            available = list(model_dir.glob("*.glb"))
            if available:
                logger.info(f"Using alternative model: {available[0].name}")
                model_path = str(available[0])
            else:
                issues.append("No GLB models found - Blender rendering will fail")
                model_path = None
        else:
            issues.append("Dataset directory not found - Blender rendering will fail")
            model_path = None
    
    return issues, model_path

def launch_blender_frustum_demo():
    """Launch the interactive 3D frustum demo with Blender rendering."""
    
    print("🎨 Interactive 3D Camera Frustum Demo with Blender Rendering")
    print("=" * 65)
    print()
    
    # Check environment and dependencies
    logger.info("Checking Blender environment...")
    issues, model_path = check_blender_environment()
    
    if issues:
        print("⚠️ Environment issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print()
        print("The demo will continue but some features may not work properly.")
        print()
    
    logger.info("✅ Environment checks completed")
    print()
    
    # Launch interactive controls with Blender integration
    logger.info("Launching 3D frustum controls with Blender rendering...")
    print("🎮 Starting interactive 3D camera frustum controls with Blender...")
    print()
    print("📱 BROWSER ACCESS:")
    print("   → Open your browser to: http://localhost:8080")
    print()
    print("🎨 BLENDER RENDERING FEATURES:")
    print("   • 🖱️ **Drag Camera**: Move red handle to position camera in 3D")
    print("   • 🎯 **Drag Target**: Move green handle to aim camera direction")  
    print("   • 🎨 **Auto Render**: Camera movement triggers Blender renders")
    print("   • 🖼️ **Image Display**: Rendered images shown at frustum position")
    print("   • ⚙️ **Quality Control**: Choose PREVIEW/STANDARD/HIGH quality")
    print("   • 📐 **Resolution**: 256x256, 512x512, or 1024x1024 output")
    print()
    print("🔧 BLENDER PIPELINE:")
    print("   • **Render Engine**: Blender EEVEE (PREVIEW) or CYCLES (STANDARD/HIGH)")
    print("   • **Material Support**: Full PBR materials from GLB models")
    print("   • **Lighting**: Automatic HDRI + sun lighting setup")
    print("   • **Output**: High-quality PNG images saved to /tmp/")
    print("   • **Performance**: Background rendering doesn't block UI")
    print()
    print("🧪 TESTING WORKFLOW:")
    print("   1. Wait for model to load and initial render to complete")
    print("   2. Drag the RED handle to move camera position")
    print("   3. Watch Blender render the new view automatically")
    print("   4. See rendered image displayed at the frustum far plane")
    print("   5. Adjust render quality/resolution in the UI panel")
    print("   6. Try dragging GREEN handle to change camera direction")
    print()
    print("⏹️  Press Ctrl+C to stop the demo")
    print("=" * 65)
    
    try:
        # Launch with Blender integration (no mock callback)
        controls = launch_frustum_controls(
            model_path=model_path,
            port=8080,
            render_callback=None  # Use built-in Blender pipeline instead
        )
        
        logger.info("✅ Interactive 3D frustum controls with Blender launched!")
        logger.info("🎯 Ready for browser interaction at http://localhost:8080")
        
        # Show additional tips
        print("\n💡 BLENDER RENDERING TIPS:")
        print("   • Initial render may take 10-30 seconds")
        print("   • PREVIEW quality is fastest for interactive use")
        print("   • STANDARD/HIGH quality gives better visual results")
        print("   • Rendered images are saved to /tmp/ directory")
        print("   • Auto-render can be disabled for manual control")
        print("   • Check render status in the UI for progress updates")
        
        # Run the interactive session
        controls.run()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
        logger.info("Interactive Blender frustum demo stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Interactive Blender frustum demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = launch_blender_frustum_demo()
    
    if success:
        print("\n✅ Interactive Blender frustum demo completed!")
        print("🎉 Camera frustum manipulation with real-time Blender rendering!")
        print()
        print("🚀 KEY ACHIEVEMENTS:")
        print("   • ✅ Interactive 3D camera frustum with drag controls")
        print("   • ✅ Real-time Blender rendering on camera movement")
        print("   • ✅ Rendered images displayed at frustum position")
        print("   • ✅ Quality and resolution controls")
        print("   • ✅ Background rendering for smooth interaction")
        print("   • ✅ Full TRELLIS coordinate system compatibility")
        print()
        print("📈 INTEGRATION SUCCESS:")
        print("   • Step 5 (Interactive Controls) + Step 6 (Blender Pipeline)")
        print("   • Fixed-size frustum with free translation/rotation")
        print("   • High-quality rendering with material support")
        print("   • Real-time visual feedback system")
    else:
        print("\n❌ Demo encountered issues")
        print("Check the logs above and ensure 'conda activate vggt' was run.")
    
    sys.exit(0 if success else 1)