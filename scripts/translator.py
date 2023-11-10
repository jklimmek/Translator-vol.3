from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from .model import Transformer


class Translator:
    """
    Translator class implementing different generation methods for the Transformer model.

    After empirical evaluation it turns out that best translations are generated
    simply using beam search with length penalty.
    """
    def __init__(self, model, tokenizer, maxlen, start_token, end_token, device):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.start_token = start_token
        self.end_token = end_token
        self.device = device


    @classmethod
    def from_config(
        cls, 
        checkpoint_path, 
        tokenizer_path, 
        maxlen=104, 
        start_token="<|startofseq|>", 
        end_token="<|endofseq|>", 
        device="cpu"
    ):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_padding(length=maxlen)
        tokenizer.enable_truncation(max_length=maxlen)
        model = Transformer.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        return cls(model, tokenizer, maxlen, start_token, end_token, device)


    @torch.no_grad()
    def translate(
        self,
        text,
        beam_size=1,
        top_k=0,
        top_p=0.0,
        temperature=1.0,
        length_penalty=0.0,
        alpha=0.0
    ):
        # Encode the text using the tokenizer and initialize the translation with the start token.
        x_tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        x = torch.tensor(x_tokens, device=self.device).unsqueeze(0)
        y = torch.zeros((1, self.maxlen), dtype=torch.long, device=self.device)
        start_token = self.tokenizer.token_to_id(self.start_token)
        y[0, 0] = start_token

        # Based on given parameters choose the appropriate generation method.
        if beam_size > 0 and top_k == 0 and top_p == 0:
            tokens = self.__beam_search(x, y, beam_size=beam_size, length_penalty=length_penalty)
        elif top_k > 0 and top_p == 0 and alpha == 0:
            tokens = self.__top_k(x, y, top_k=top_k, temperature=temperature)
        elif top_k == 0 and top_p > 0 and alpha == 0:
            tokens = self.__top_p(x, y, top_p=top_p, temperature=temperature)
        elif top_k > 0 and top_p == 0 and alpha > 0:
            tokens = self.__contrastive_search(x, y, top_k=top_k, alpha=alpha)

        # Decode the tokens into text and return it.
        translated = self.tokenizer.decode(tokens)
        return translated
    

    def __beam_search(self, x, y, beam_size, length_penalty):
        # Initialize beam candidates with the initial translation and a score of 0.0.
        beam_candidates = [{'translation': y, 'score': 0.0}]

        for i in range(1, self.maxlen):
            # Store the candidates for the next beam in this list.
            next_beam_candidates = []

            # Iterate over the current beam candidates.
            for candidate in beam_candidates:
                # Extract the partial translation from the candidate.
                partial_translation = candidate['translation']

                # Generate logits for the next token in the sequence using the model.
                logits = self.model(src=x, tgt=partial_translation)
                probabilities = F.softmax(logits[0, i - 1], dim=-1)

                # Select top tokens based on probabilities for beam expansion.
                if beam_size > 1:
                    top_tokens = probabilities.topk(beam_size)[1].squeeze()
                else:
                    # If beam size is 1, directly choose the token with the highest probability.
                    # This is equivalent to greedy search.
                    top_token = probabilities.argmax()
                    top_tokens = torch.tensor([top_token.item()], device=self.device)

                # Create new beam candidates by expanding with top tokens.
                for token in top_tokens:
                    new_translation = partial_translation.clone()
                    new_translation[:, i] = token.item()

                    # Update the score based on the log probability of the selected token.
                    new_score = candidate['score'] + torch.log(probabilities[token])

                    # Apply length penalty to the score.
                    penalty = ((5 + i) / 6) ** length_penalty
                    new_score /= penalty

                    # Add the new candidate to the list for the next beam.
                    next_beam_candidates.append({'translation': new_translation, 'score': new_score})

            # Sort the next beam candidates based on scores in descending order.
            next_beam_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Select the top candidates to form the new beam.
            beam_candidates = next_beam_candidates[:beam_size]

            # Check if any of the top candidates end with the end token, and break the loop if so.
            if all(self.__check_end_token(candidate['translation'][:, i].item()) for candidate in beam_candidates):
                break

        # Get the best translation from the top candidate in the final beam.
        best_translation = beam_candidates[0]['translation']

        # Return the translation.
        return best_translation[0].cpu().numpy()


    def __top_k(self, x, y, top_k, temperature):
        for i in range(1, self.maxlen):

            # Generate logits for the next token using the model.
            logits = self.model(src=x, tgt=y)
            logits = logits[0, i - 1] / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Select top-k tokens based on probabilities.
            top_k_values, top_k_indices = probabilities.topk(top_k)
            top_k_values = top_k_values / torch.sum(top_k_values)

            # Sample a token from the top-k distribution.
            next_token_index = torch.multinomial(top_k_values, num_samples=1)
            next_token = top_k_indices[next_token_index].item()

            # Update the translation with the new token.
            y[:, i] = next_token

            # Stop generating if the end_token is selected.
            if self.__check_end_token(next_token):
                break
        
        # Return the translation.
        return y[0].cpu().numpy() 


    def __top_p(self, x, y, top_p, temperature):
        for i in range(1, self.maxlen):

            # Generate logits for the next token using the model.
            logits = self.model(src=x, tgt=y)
            logits = logits[0, i - 1] / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Select top-p tokens based on cumulative probabilities.
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_indices_to_remove = cumulative_probabilities > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            sorted_probabilities[sorted_indices_to_remove] = 0
            sorted_probabilities = sorted_probabilities / torch.sum(sorted_probabilities)

            # Sample a token from the top-p distribution.
            next_token_index = torch.multinomial(sorted_probabilities, num_samples=1)
            next_token = sorted_indices[next_token_index].item()

            # Update the translation with the new token.
            y[:, i] = next_token

            # Stop generating if the end_token is selected.
            if self.__check_end_token(next_token):
                break
        
        # Return the translation.
        return y[0].cpu().numpy()


    def __contrastive_search(self, x, y, top_k, alpha):
        
        # Initialize the list of word embeddings with the embedding of the first token.
        word_embeddings_matrix = self.model.xformer.get_submodule("decoders.0.pose_encoding.word_embeddings")
        word_embeddings = [word_embeddings_matrix(y[:, 0]).squeeze(0)]

        for i in range(1, self.maxlen):

            # Generate logits for the next token using the model.
            logits = self.model(src=x, tgt=y)
            probabilities = F.softmax(logits[0, i - 1], dim=-1)

            # Filter logits to keep only the top-k most probable tokens.
            confidences, indices = probabilities.topk(top_k)

            # Calculate degeneration penalty for each token.
            scores = []
            for conf, ind in zip(confidences, indices):
                degeneration_penalty = max([torch.cosine_similarity(word_embeddings_matrix(ind), word, dim=-1) for word in word_embeddings])
                scores.append((1 - alpha) * conf - alpha * degeneration_penalty)
            
            # Select the token with the highest score.
            index = scores.index(max(scores))
            next_token = indices[index].item()

            # Update the translation with the new token.
            y[:, i] = next_token

            # Add new token's embedding to the list of embeddings.
            word_embeddings.append(word_embeddings_matrix(torch.tensor(next_token)).squeeze(0))

            # If the token is end_token stop generating.
            if self.__check_end_token(next_token):
                break
        
        # Return the translation.
        return y[0].cpu().numpy()


    def __check_end_token(self, token):
        # Check if the token is the end_token.
        return token == self.tokenizer.token_to_id(self.end_token)