import streamlit as st
from scripts.translator import Translator


TOKENIZER_PATH = "tokenizers/uncased-32000.json"
MODEL_PATH = "checkpoints/Transformer_60M/epoch=02-train_loss=1.9006-dev_loss=1.8697.ckpt"


class Slider:
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


def app():
    st.set_page_config(page_title="French-English Translator", layout="wide", page_icon="🇫🇷")
    # st.title("French-English Translator")
    col1, col2 = st.columns((1, 2))
    with col1:
        st.subheader("Adjust generation parameterts")

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

        max_tokens = max_tokens.value
        beam_size = beam_size.value
        temperature = temperature.value
        top_p = top_p.value
        top_k = top_k.value
        length_penalty = length_penalty.value
        alpha = alpha.value

        if beam_size > 0 and top_k == 0 and top_p == 0:
            message = f"Beam search with beam_size = {beam_size} and length penalty = {length_penalty}"
        elif top_k > 0 and top_p == 0 and alpha == 0:
            message = f"Top-k sampling with k = {top_k} and temperature = {temperature}"
        elif top_k == 0 and top_p > 0 and alpha == 0:
            message = f"Top-p sampling with p = {top_p} and temperature = {temperature}"
        elif top_k > 0 and top_p == 0 and alpha > 0:
            message = f"Contrastive sampling with k = {top_k}, alpha = {alpha} and temperature = {temperature}"
        else:
            message = "No generation method selected"

        if message != "No generation method selected":
            st.info(message)
        else:
            st.warning(message, icon="⚠️")

    with col2:
        st.subheader("Enter French text to translate")
        text = st.text_area(
            "French text", 
            placeholder = "Enter French text to translate", 
            height = 300, 
            label_visibility = "hidden"
        )

        translator = Translator.from_config(
            checkpoint_path = MODEL_PATH,
            tokenizer_path = TOKENIZER_PATH,
        )

        if text:
            # with st.spinner("Translating..."):
            #    translation = "sample translation to be displayed."
            
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
            if translation is not None:
                st.success("Translation Successful")
                st.text_area("Translated Text", translation, height=300)
            else:
                st.error("Translation Failed:")
                st.warning("Please check your input and try again.")

        

if __name__ == "__main__":
    app()