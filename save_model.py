from transformers import TFBertForSequenceClassification

# Paths to adjust according to your directory
MODEL_PATH = "C:\Users\sudar\Desktop\chatbot\models\chatbot_final_model.keras"  # Adjust if this path is incorrect
OUTPUT_DIR = "C:\Users\sudar\Desktop\chatbot\models"  # Directory to save Hugging Face-compatible files

try:
    # Load the existing trained model
    print("🔄 Loading model...")
    bert_model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, from_tf=True)

    # Save the model in Hugging Face format
    print("💾 Saving model in Hugging Face format...")
    bert_model.save_pretrained(OUTPUT_DIR)
    print("✅ Model saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
