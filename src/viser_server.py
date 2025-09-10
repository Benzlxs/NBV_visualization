"""Basic Viser server implementation for 3D visualization."""

import viser
import time
from typing import Optional


class BasicViserServer:
    """A basic Viser server that can display 3D objects."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize the Viser server.
        
        Args:
            host: Host address to bind to
            port: Port number to use
        """
        self.host = host
        self.port = port
        self.server: Optional[viser.ViserServer] = None
    
    def start(self):
        """Start the Viser server."""
        self.server = viser.ViserServer(host=self.host, port=self.port)
        print(f"Viser server started at http://localhost:{self.port}")
        return self.server
    
    def add_coordinate_frame(self, name: str = "/world", axes_length: float = 0.5):
        """Add a coordinate frame to the scene.
        
        Args:
            name: Name of the coordinate frame
            axes_length: Length of the axes
        """
        if self.server is None:
            raise RuntimeError("Server not started. Call start() first.")
        
        self.server.scene.add_frame(
            name=name,
            axes_length=axes_length
        )
    
    def keep_alive(self):
        """Keep the server running."""
        if self.server is None:
            raise RuntimeError("Server not started. Call start() first.")
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
    
    def stop(self):
        """Stop the server."""
        if self.server is not None:
            # Viser servers are cleaned up automatically
            self.server = None
            print("Server stopped.")


def main():
    """Example usage of the BasicViserServer."""
    server = BasicViserServer()
    server.start()
    server.add_coordinate_frame()
    print("Press Ctrl+C to stop the server")
    server.keep_alive()


if __name__ == "__main__":
    main()