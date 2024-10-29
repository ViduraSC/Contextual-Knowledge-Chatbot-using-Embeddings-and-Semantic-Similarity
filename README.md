
# Contextual Knowledge Chatbot using Embeddings and Semantic Similarity

## Overview
This project implements a contextual knowledge chatbot that leverages embeddings and semantic similarity techniques to provide relevant responses based on user queries. Utilizing the Sentence Transformers library, the chatbot encodes user input and compares it with a pre-existing knowledge base to find the most relevant documents. The system enhances user interaction by providing context-aware answers, making it suitable for various applications in customer support and information retrieval.

## Techniques Used
- **Sentence Transformers**: For encoding user queries and documents into high-dimensional embeddings.
- **Cosine Similarity**: To calculate the similarity between user queries and the knowledge base, allowing for efficient retrieval of relevant information.
- **Flask**: A lightweight web framework used to create the web interface for the chatbot, enabling seamless interaction.
- **HTML/CSS/JavaScript**: For the frontend design of the chatbot interface, ensuring a user-friendly experience.
- **NumPy**: For handling numerical operations, particularly in managing embeddings.
- **JSON**: For storing and loading the knowledge base and embeddings.

## Folder Structure
Contextual-Knowledge-Chatbot-using-Embeddings-and-Semantic-Similarity/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── knowledge_base.json         # Knowledge base containing documents
├── embeddings.npy              # Precomputed document embeddings
├── templates/
│   └── index.html             # HTML template for the chatbot interface
└── static/
    ├── logo.png               # Company logo
    ├── user-profile-pic.png    # User profile picture
    └── bot-profile-pic.png     # Bot profile picture

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ViduraSC/Contextual-Knowledge-Chatbot-using-Embeddings-and-Semantic-Similarity.git
   cd Contextual-Knowledge-Chatbot-using-Embeddings-and-Semantic-Similarity
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000` to interact with the chatbot.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

