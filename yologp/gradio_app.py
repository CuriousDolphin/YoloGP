import gradio as gr

from frame_extractor_gradio_app import frame_ext_interface
from inference_gradio_app import inference_interface


tabbed_interface = gr.TabbedInterface(
    interface_list=[inference_interface, frame_ext_interface],
    tab_names=["Inference", "Extract Frame"],
)

if __name__ == "__main__":
    tabbed_interface.queue(max_size=10).launch(server_name="0.0.0.0")
