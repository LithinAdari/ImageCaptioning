import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from collections import Counter
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==============================================================================
# 1. Hyperparameters & Settings
# ==============================================================================
# Kaggle default path for datasets (assuming you use: 'Add Data' -> Flickr8k)
# This path might need to be adjusted based on the specific Kaggle dataset used:
# e.g., '/kaggle/input/flickr8k/Images' and '/kaggle/input/flickr8k/captions.txt'
DATA_DIR = "/kaggle/input/flickr8k" 
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE_G = 3e-4
LEARNING_RATE_D = 5e-5 # Lower LR for Discriminator to prevent it from overpowering
MAX_SEQ_LEN = 50
VOCAB_Threshold = 5 # Minimum frequency for a word to be in the vocab.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 2. Vocabulary creation
# ==============================================================================
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize_en(text):
        return text.lower().strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize_en(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize_en(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# ==============================================================================
# 3. Dataset & DataLoader Definition
# ==============================================================================
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.transform = transform
        
        # Read the captions file (Assuming format: image_name,caption)
        self.imgs = []
        self.captions = []
        
        # Read file, skip header if present
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Skip header if it is "image,caption"
                if "image" in lines[0].lower() and "caption" in lines[0].lower():
                    lines = lines[1:]
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    # Flickr8k captions.txt usually has "image_name,caption"
                    # old flickr8k sometimes has "image_name#0\tcaption"
                    if '\t' in line:
                         parts = line.split('\t', 1)
                         img_id = parts[0].split('#')[0] # remove #0, #1
                         caption = parts[1]
                    else:
                         parts = line.split(',', 1)
                         if len(parts) != 2: continue
                         img_id = parts[0]
                         caption = parts[1]
                         
                    self.imgs.append(img_id)
                    self.captions.append(caption)
        except Exception as e:
            print(f"Error reading captions: {e}")
            print(f"Make sure {CAPTION_FILE} exists and is right format.")
            
        print(f"Loaded {len(self.imgs)} image-caption pairs.")
        # Initialize and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.image_dir, img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
             # Fallback dummy image if file randomly fails
             img = Image.new('RGB', (224, 224), color = 'black')
             
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx, max_seq_len):
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        
        # Pad and Truncate logic
        batch_size = len(targets)
        # Find the max sequence length in this batch, capped at max_seq_len
        max_batch_seq_len = min(max(len(t) for t in targets), self.max_seq_len)
        
        padded_targets = torch.full((batch_size, max_batch_seq_len), self.pad_idx, dtype=torch.long)
        
        for i, target in enumerate(targets):
            length = min(len(target), max_batch_seq_len)
            padded_targets[i, :length] = target[:length]
            
        return imgs, padded_targets

def get_loaders(root_folder, annotation_file, transform, batch_size=32, num_workers=2, pin_memory=True):
    dataset = Flickr8kDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # Split dataset into train (80%), val (10%), test (10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # We use a fixed seed to ensure splits are consistent across runs
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, max_seq_len=MAX_SEQ_LEN),
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, max_seq_len=MAX_SEQ_LEN),
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx, max_seq_len=MAX_SEQ_LEN),
    )
    return train_loader, val_loader, test_loader, dataset

# Image transformations for ViT (224x224 and standard ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ==============================================================================
# 4. Model Architectures (From our implementation plan)
# ==============================================================================
class ViTEncoder(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Freeze ViT feature extractor to speed up training drastically
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.d_model = d_model
        
    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x

class Generator(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.encoder = ViTEncoder(d_model=d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, images, captions):
        memory = self.encoder(images) 
        seq_len = captions.size(1)
        tgt = self.embedding(captions) + self.pos_encoder[:, :seq_len, :] 
        
        # Causal mask for decoder
        caption_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        out = self.decoder(tgt, memory, tgt_mask=caption_mask) 
        logits = self.fc_out(out) 
        return logits

class Discriminator(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.image_encoder = ViTEncoder(d_model=d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, images, captions):
        B, seq_len = captions.size()
        memory = self.image_encoder(images) 
        emb = self.embedding(captions) + self.pos_encoder[:, :seq_len, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tgt = torch.cat((cls_tokens, emb), dim=1) 
        
        out = self.transformer_decoder(tgt, memory) 
        cls_out = out[:, 0, :] 
        validity = self.fc_out(cls_out) 
        return validity

# ==============================================================================
# 5. Helper Function: Sample from Generator (to feed to Discriminator)
# ==============================================================================
def sample_captions(generator, images, max_len=MAX_SEQ_LEN, start_idx=1, end_idx=2):
    """
    Greedy decoding to generate captions for a batch of images to pass to Discriminator.
    """
    B = images.size(0)
    memory = generator.encoder(images)
    
    # Initialize target tensor with <START> token (idx 1 usually)
    tgt = torch.full((B, 1), start_idx, dtype=torch.long, device=images.device)
    
    for _ in range(max_len - 1):
        seq_len = tgt.size(1)
        tgt_emb = generator.embedding(tgt) + generator.pos_encoder[:, :seq_len, :]
        caption_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        
        out = generator.decoder(tgt_emb, memory, tgt_mask=caption_mask)
        logits = generator.fc_out(out[:, -1, :]) # Predict next token from the last generated time step
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
    return tgt

def evaluate(generator, discriminator, loader, criterion_CE, criterion_GAN, vocab):
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_d_loss = 0
    total_bleu_1 = 0
    total_bleu_4 = 0
    samples_count = 0
    
    smoothie = SmoothingFunction().method4
    
    with torch.no_grad():
        for imgs, captions in loader:
            imgs, captions = imgs.to(device), captions.to(device)
            B = imgs.size(0)
            
            # --- Losses ---
            real_labels = torch.ones(B, 1).to(device)
            fake_labels = torch.zeros(B, 1).to(device)
            
            # Discriminator loss
            real_validity = discriminator(imgs, captions)
            d_real_loss = criterion_GAN(real_validity, real_labels)
            
            sampled_fake = sample_captions(generator, imgs)
            fake_validity = discriminator(imgs, sampled_fake)
            d_fake_loss = criterion_GAN(fake_validity, fake_labels)
            
            total_d_loss += (d_real_loss + d_fake_loss).item()
            
            # Generator loss
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            outputs = generator(imgs, input_captions)
            ce_loss = criterion_CE(outputs.reshape(-1, len(vocab)), target_captions.reshape(-1))
            total_g_loss += ce_loss.item()
            
            # --- Metrics (BLEU Score) ---
            for i in range(B):
                # Convert target sequence to words
                ref_tokens = []
                for idx in target_captions[i]:
                    if idx.item() == vocab.stoi["<END>"] or idx.item() == vocab.stoi["<PAD>"]:
                        break
                    ref_tokens.append(vocab.itos[idx.item()])
                
                # Convert generated sequence to words
                gen_tokens = []
                for idx in sampled_fake[i][1:]: # Skip <START>
                    if idx.item() == vocab.stoi["<END>"] or idx.item() == vocab.stoi["<PAD>"]:
                        break
                    gen_tokens.append(vocab.itos[idx.item()])
                
                if len(ref_tokens) > 0 and len(gen_tokens) > 0:
                    bleu_1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                    bleu_4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                    
                    total_bleu_1 += bleu_1
                    total_bleu_4 += bleu_4
                    samples_count += 1
                    
    generator.train()
    discriminator.train()
    
    avg_d_loss = total_d_loss / len(loader)
    avg_g_loss = total_g_loss / len(loader)
    avg_bleu_1 = total_bleu_1 / max(1, samples_count)
    avg_bleu_4 = total_bleu_4 / max(1, samples_count)
    
    return avg_d_loss, avg_g_loss, avg_bleu_1, avg_bleu_4

# ==============================================================================
# 6. Main Training Loop
# ==============================================================================
def train_gan():
    print("Setting up DataLoaders...")
    # NOTE: In Kaggle, ensure `DATA_DIR` contains Images folder and captions.txt
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(CAPTION_FILE):
        print(f"Warning: Data paths {IMAGE_DIR} / {CAPTION_FILE} not found. Please attach the dataset.")
        return

    train_loader, val_loader, test_loader, dataset = get_loaders(
        root_folder=IMAGE_DIR,
        annotation_file=CAPTION_FILE,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    vocab_size = len(dataset.vocab)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Train split: {len(train_loader.dataset)} samples")
    print(f"Val split: {len(val_loader.dataset)} samples")
    print(f"Test split: {len(test_loader.dataset)} samples")

    generator = Generator(vocab_size=vocab_size, num_layers=4).to(device)
    discriminator = Discriminator(vocab_size=vocab_size, num_layers=4).to(device)

    # Note: Generator uses CrossEntropy for language modeling, but we combine it with adversarial loss
    criterion_CE = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    criterion_GAN = nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D)
    
    # Scheduler to reduce learning rate if validation loss plateaus
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=2)
    
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # Weighting the GAN loss against standard CE loss (Language Modeling)
    alpha = 0.01  # Reduced alpha. If D loss goes to 0 rapidly, it ruins G training.

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        for batch_idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # -----------------------------------------------------------
            # Train Discriminator
            # -----------------------------------------------------------
            # Real labels = 1, Fake labels = 0
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            with torch.cuda.amp.autocast():
                # Real forward pass
                real_validity = discriminator(imgs, captions)
                d_real_loss = criterion_GAN(real_validity, real_labels)
                
                # Fake forward pass (use generator to get completely fake sequence)
                # To prevent graph holding, we detach the generated captions 
                with torch.no_grad():
                    sampled_fake_captions = sample_captions(generator, imgs)
                    
                fake_validity = discriminator(imgs, sampled_fake_captions)
                d_fake_loss = criterion_GAN(fake_validity, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                
            optimizer_D.zero_grad()
            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
            d_epoch_loss += d_loss.item()

            # -----------------------------------------------------------
            # Train Generator
            # -----------------------------------------------------------
            with torch.cuda.amp.autocast():
                # 1. Standard Language Modeling Loss (Cross-Entropy)
                # The input to generator is captions shifted by 1 (we don't pass the last target token as input)
                # The target is the original captions
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                
                outputs = generator(imgs, input_captions) 
                # outputs: [B, seq_len-1, vocab_size]
                
                # Reshape for CE Loss: [B * seq_len, vocab_size], [B * seq_len]
                ce_loss = criterion_CE(outputs.reshape(-1, vocab_size), target_captions.reshape(-1))
                
                # 2. Adversarial Loss
                # The generator tries to trick D into mapping fakes to 1 (real)
                sampled_fake_captions_for_G_loss = sample_captions(generator, imgs)
                fake_validity_for_G = discriminator(imgs, sampled_fake_captions_for_G_loss)
                g_adv_loss = criterion_GAN(fake_validity_for_G, real_labels) # We want D to output 1 here
                
                # Total G Loss
                g_loss = ce_loss + (alpha * g_adv_loss)
            
            optimizer_G.zero_grad()
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            g_epoch_loss += g_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f} (CE: {ce_loss.item():.4f}, ADV: {g_adv_loss.item():.4f})")
                
        end_time = time.time()
        print(f"--- Epoch {epoch+1} complete in {end_time - start_time:.2f} seconds ---")
        print(f"Train - Avg D_Loss: {d_epoch_loss/len(train_loader):.4f}, Avg G_Loss: {g_epoch_loss/len(train_loader):.4f}")
        
        # Validation Evaluation
        val_d_loss, val_g_loss, val_b1, val_b4 = evaluate(
            generator, discriminator, val_loader, criterion_CE, criterion_GAN, dataset.vocab
        )
        
        # Step the scheduler based on CE loss
        scheduler_G.step(val_g_loss)
        
        print(f"Validation - D_Loss: {val_d_loss:.4f} | G_Loss(CE): {val_g_loss:.4f} | BLEU-1: {val_b1:.4f} | BLEU-4: {val_b4:.4f}")
        print("-" * 80)
        
    print("=================== Training Complete ===================")
    print("Running Final Evaluation on Test Set...")
    test_d_loss, test_g_loss, test_b1, test_b4 = evaluate(
        generator, discriminator, test_loader, criterion_CE, criterion_GAN, dataset.vocab
    )
    print(f"TEST RESULTS:")
    print(f"Discriminator Loss: {test_d_loss:.4f}")
    print(f"Generator CE Loss:  {test_g_loss:.4f}")
    print(f"BLEU-1 Score:       {test_b1:.4f}")
    print(f"BLEU-4 Score:       {test_b4:.4f}")
    
    # Save the models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Models saved to 'generator.pth' and 'discriminator.pth'")

if __name__ == "__main__":
    train_gan()
