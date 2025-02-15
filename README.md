# Mental Health Chatbot

This is an AI-powered mental health chatbot built using TensorFlow, Keras, and BERT. The chatbot is designed to provide mental health support by recognizing user inputs and generating appropriate responses.

## Features
- Uses a pre-trained BERT model for natural language understanding.
- Trained on a custom mental health dataset.
- Provides mental health support and general well-being advice.
- Can be integrated with APIs for extended functionality.

## Technologies Used
- Python
- TensorFlow
- Keras
- BERT (Bidirectional Encoder Representations from Transformers)
- Natural Language Processing (NLP)
- Pandas (for data handling)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/mental-health-chatbot.git
   cd mental-health-chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows use: chatbot_env\Scripts\activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The chatbot uses a dataset located at:
- `dataset/mentalhealth.csv`
- `dataset/mentalhealth.json`

Ensure that the dataset is preprocessed before training the model.

## Training the Model
Run the following command to train the chatbot:
```bash
python train.py
```

## Running the Chatbot
To start the chatbot, run:
```bash
python chatbot.py
```

## API Integration
The chatbot can be extended using APIs such as Google Gemini and LLaMA for fallback responses.

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is licensed under the MIT License.

