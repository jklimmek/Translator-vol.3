import streamlit as st
from scripts.translator import Translator


TOKENIZER_PATH = "tokenizers/uncased-32000.json"
MODEL_PATH = "checkpoints/Transformer_60M/epoch=02-train_loss=1.9006-dev_loss=1.8697.ckpt"


class Slider:
    """
    A class to create a slider and a numeric input that are synced together.

    Syncing is done by referencing the session state of the slider
    to the session state of the numeric input and vice versa.
    """
    def __init__(self, name, min_value, max_value, step, default, index=0, columns=(3, 1)):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.default = default
        self.index = index
        self.columns = columns

    @property
    def value(self):
        col1, col2 = st.columns(self.columns)
        with col1:
            slider_val = self.get_slider_value()
        with col2:
            numeric_val = self.get_numeric_value()
        assert slider_val == numeric_val, "Slider and numeric values are not equal"
        return slider_val
    
    def get_slider_value(self):
        return st.slider(
            self.name, 
            self.min_value, 
            self.max_value, 
            self.default, 
            self.step, 
            key=f"slider_{self.index}",
            on_change=self.update_numeric
        )
    
    def get_numeric_value(self):
        return st.number_input(
            self.name,
            self.min_value, 
            self.max_value, 
            self.default, 
            self.step, 
            key=f"numeric_{self.index}",
            on_change=self.update_slider,
            label_visibility="hidden"
        )
    
    def update_slider(self):
        st.session_state[f"slider_{self.index}"] = st.session_state[f"numeric_{self.index}"]

    def update_numeric(self):
        st.session_state[f"numeric_{self.index}"] = st.session_state[f"slider_{self.index}"]


@st.cache_resource()
def load_model():
    """
    Load model and tokenizer from checkpoint and tokenizer path.
    Decorator ensures that the model is only loaded once.
    """
    translator = Translator.from_config(
        checkpoint_path = MODEL_PATH,
        tokenizer_path = TOKENIZER_PATH,
    )
    return translator


def app():
    """
    Streamlit app to translate French text to English.
    """

    # Set page title, layout and favicon.
    st.set_page_config(page_title="French-English Translator", layout="wide", page_icon="🇫🇷")
    col1, col2 = st.columns((1, 2))

    # Load model and tokenizer.
    translator = load_model()

    with col1:
        st.subheader("Adjust generation parameterts")

        # Initialize sliders with parameters for generation.
        max_tokens = Slider(
            name = "max_tokens",
            min_value = 0,
            max_value = 104,
            step = 1,
            default = 104,
            index = 0
        )

        beam_size = Slider(
            name = "beam_size",
            min_value = 1,
            max_value = 5,
            step = 1,
            default = 1,
            index = 1
        )

        temperature = Slider(
            name = "temperature",
            min_value = 0.0,
            max_value = 1.0,
            step = 0.01,
            default = 1.0,
            index = 2
        )
        top_p = Slider(
            name = "top_p",
            min_value = 0.0,
            max_value = 1.0,
            step = 0.01,
            default = 0.0,
            index = 3
        )
        top_k = Slider(
            name = "top_k",
            min_value = 0,
            max_value = 50,
            step = 1,
            default = 0,
            index = 4
        )

        length_penalty = Slider(
            name = "length_penalty",
            min_value = 0.0,
            max_value = 1.0,
            step = 0.01,
            default = 0.0,
            index = 5
        )

        alpha = Slider(
            name = "alpha",
            min_value = 0.0,
            max_value = 1.0,
            step = 0.01,
            default = 0.0,
            index = 6
        )

        # Display sliders.
        max_tokens = max_tokens.value
        beam_size = beam_size.value
        temperature = temperature.value
        top_p = top_p.value
        top_k = top_k.value
        length_penalty = length_penalty.value
        alpha = alpha.value

        # Display generation method.
        if beam_size > 0 and top_k == 0 and top_p == 0:
            message = f"Beam search with beam_size = {beam_size} and length penalty = {length_penalty}"
        elif top_k > 0 and top_p == 0 and alpha == 0:
            message = f"Top-k sampling with k = {top_k} and temperature = {temperature}"
        elif top_k == 0 and top_p > 0 and alpha == 0:
            message = f"Top-p sampling with p = {top_p} and temperature = {temperature}"
        elif top_k > 0 and top_p == 0 and alpha > 0:
            message = f"Contrastive search with k = {top_k}, alpha = {alpha} and temperature = {temperature}"
        else:
            message = "No generation method selected"

        # Display message and warning if no valid generation method is selected.
        # E.g. if top_k and top_p are both > 0
        if message != "No generation method selected":
            st.info(message)
        else:
            st.warning(message, icon="⚠️")

    with col2:
        st.subheader("Enter French text to translate")

        # Display text area to enter French text.
        text = st.text_area(
            "French text", 
            placeholder = "Enter French text to translate", 
            height = 300, 
            label_visibility = "hidden"
        )

        # Translate text.
        if text:
            st.subheader("English translation")
            with st.spinner("Translating..."):
                translation = translator.translate(
                    text = text,
                    max_tokens = max_tokens,
                    beam_size = beam_size,
                    temperature = temperature,
                    top_p = top_p,
                    top_k = top_k,
                    length_penalty = length_penalty,
                    alpha = alpha
                )

            # Display translated text.
            st.text_area(
                "Translated Text", 
                translation, 
                height=300, 
                label_visibility="hidden"
            )


if __name__ == "__main__":
    app()