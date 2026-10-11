"""Launch the RevCT chest-CT demo."""

from demo.backend import DemoBackend
from demo.ui import build_demo, demo_launch_kwargs

backend = DemoBackend()
backend.prepare_for_host()
demo = build_demo(backend)

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        max_file_size="512mb",
        **demo_launch_kwargs(),
    )
