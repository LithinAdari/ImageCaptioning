import torch
import torch.nn as nn
import torchvision.models as models

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder to extract visual features from images.
    Uses a pretrained ViT from torchvision and extracts the sequence of patch embeddings.
    """
    def __init__(self, d_model=768):
        super().__init__()
        # Load a pretrained ViT-B/16 model
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.d_model = d_model
        
    def forward(self, x):
        # x shape: [Batch_Size, 3, 224, 224]
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Pass through the ViT encoder to get contextualized patch embeddings
        x = self.vit.encoder(x)
        # return shape: [Batch_Size, 197, 768] (1 CLS token + 196 patch tokens)
        return x

class Generator(nn.Module):
    """
    Generator model: ViT Encoder + Transformer Decoder.
    Generates captions based on the visual input.
    """
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=4, max_seq_len=50):
        super().__init__()
        self.encoder = ViTEncoder(d_model=d_model)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, images, captions, caption_mask=None):
        """
        images: [B, 3, 224, 224]
        captions: [B, seq_len] 
        """
        # 1. Encode image to acquire visual features
        memory = self.encoder(images)  # [B, 197, d_model]
        
        # 2. Embed captions
        seq_len = captions.size(1)
        tgt = self.embedding(captions) + self.pos_encoder[:, :seq_len, :] # [B, seq_len, d_model]
        
        # 3. Decode caption using visual features as memory
        if caption_mask is None:
            # Causal mask to ensure decoder only attends to previous tokens
            caption_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
            
        out = self.decoder(tgt, memory, tgt_mask=caption_mask) # [B, seq_len, d_model]
        
        # 4. Generate token logits
        logits = self.fc_out(out) # [B, seq_len, vocab_size]
        return logits

class Discriminator(nn.Module):
    """
    Discriminator model: Transformer Decoder checking Image + Caption coherence.
    Predicts if the caption is real or fake for the given image.
    """
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=4, max_seq_len=50):
        super().__init__()
        self.image_encoder = ViTEncoder(d_model=d_model)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Using a transformer decoder where image features act as memory
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, images, captions):
        """
        images: [B, 3, 224, 224]
        captions: [B, seq_len]
        """
        B, seq_len = captions.size()
        
        # 1. Extract visual features
        memory = self.image_encoder(images) # [B, 197, d_model]
        
        # 2. Embed sequence and prepend a classification token
        emb = self.embedding(captions) + self.pos_encoder[:, :seq_len, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tgt = torch.cat((cls_tokens, emb), dim=1) # [B, seq_len + 1, d_model]
        
        # 3. Attend caption to image features
        out = self.transformer_decoder(tgt, memory) # [B, seq_len + 1, d_model]
        
        # 4. Gather the CLS token output representing the whole image-caption pair
        cls_out = out[:, 0, :] # [B, d_model]
        
        # 5. Output real/fake logit
        validity = self.fc_out(cls_out) # [B, 1]
        return validity

if __name__ == "__main__":
    print("Testing Image Captioning GAN models...")
    batch_size = 2
    vocab_size = 1000
    seq_len = 20
    
    # Dummy image and caption data
    # Standard ImageNet normalization shape
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Initializing Generator...")
    generator = Generator(vocab_size=vocab_size, num_layers=2) # Using 2 layers for fast testing
    
    print("Initializing Discriminator...")
    discriminator = Discriminator(vocab_size=vocab_size, num_layers=2)
    
    print("--- Running Forward Pass (Generator) ---")
    gen_out = generator(dummy_images, dummy_captions)
    print(f"Generator output shape: {gen_out.shape} (Expected: [{batch_size}, {seq_len}, {vocab_size}])")
    
    print("--- Running Forward Pass (Discriminator) ---")
    disc_out = discriminator(dummy_images, dummy_captions)
    print(f"Discriminator output shape: {disc_out.shape} (Expected: [{batch_size}, 1])")
    
    print("Successfully implemented and verified the architecture!")
