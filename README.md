# ALU University Admissions Chatbot

## 🎥 **VIDEO DEMO** - Click to Watch!
[![ALU Chatbot Demo](https://img.shields.io/badge/Watch%20Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/LtUOlQm8Z3A?si=yvrCzfo09KEtj8IV)
> **Quick Access**: Watch the 7-8 minute demonstration of the chatbot in action!

---

Hey! This is my chatbot project for ALU University admissions. I built this to help students get answers about applying to ALU. It uses some cool AI stuff - basically a neural network that can chat about university stuff. I made it specifically for ALU since that's where I study.

## What This Project Does

I created this chatbot to help students who want to apply to ALU. It can answer questions about:
- What programs ALU offers
- How much tuition costs
- How to apply
- Entry requirements
- And other admissions stuff

I built two versions - a simple one that works fast, and a more advanced one using GPT-2 that gives better answers but needs more computer power.

## What I Learned Building This

### The Dataset
I used a dataset from Hugging Face that has conversational data for university admissions. I picked one that was specifically about educational chatbots and fine-tuned it for ALU admissions. The dataset had conversations about programs, costs, requirements, and application processes that I adapted for ALU.

### Training the Models
For the simple version, I used a basic neural network that classifies what the user is asking about and gives a pre-written response. It works pretty well and doesn't need much computer power.

For the advanced version, I tried fine-tuning GPT-2. It took forever to train and needed a good computer, but it can generate more natural-sounding responses. I experimented with different settings to make it work better.

### Testing Everything
I tested both versions with questions that real students might ask. The simple one is faster and more reliable, but the GPT-2 one sometimes gives more detailed answers.

## Cool Features I Added

- **Two Ways to Chat**: You can use either a simple bot that gives quick answers, or a smarter one that understands context better
- **ALU-Specific Answers**: All the responses are about ALU - programs, costs, applications, etc.
- **Nice Web Interface**: I made a pretty chat interface that works on phones and computers
- **Real-Time Chat**: You can have a conversation and it responds right away
- **Testing Tools**: I built ways to check if the bot gives good answers

## Why I Built This (For My Assignment)

I made this chatbot for my machine learning class. The assignment was to create an AI system that could help with a real-world problem. I chose ALU admissions because I go to ALU and I know how confusing applying to university can be.

### What I Did Right
- **Real Problem**: Many students struggle with ALU's admissions process, so a chatbot could really help
- **Good Dataset**: I used a Hugging Face dataset with real conversational data for educational chatbots
- **Two Approaches**: I tried both a simple method and a more advanced one to show I understand different techniques
- **Working Website**: You can actually chat with the bot through a nice web interface
- **Testing**: I built ways to check if the bot gives good answers

### Challenges I Faced
- Training the GPT-2 model took forever and needed a lot of computer power
- Figuring out the right settings for the neural network was tricky
- Making sure the bot gives accurate information about ALU was important

## How to Run This Project

### First, Install Stuff
You'll need Python and some libraries. Here's what I used:

```bash
# Make a virtual environment (this keeps things organized)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or on Windows: venv\Scripts\activate

# Install all the libraries
pip install transformers torch accelerate datasets evaluate nltk scikit-learn fastapi uvicorn
```

### Train the Chatbot
I made two versions - pick whichever you want:

**Easy Version (Recommended):**
```bash
cd src
python train.py
```

**Advanced Version (Takes longer):**
```bash
# Use the Jupyter notebook for the full process
jupyter notebook ALU_Chatbot_Project.ipynb
```

### Start the Chatbot
```bash
cd src
python app.py
```

Then open your browser and go to http://localhost:8000 - you can chat with the bot!

## How the Chatbot Works

### The Simple Version
I used a basic neural network that looks at the words in your message and figures out what you're asking about. Then it gives you a pre-written answer. It's like having a smart lookup table - fast and reliable, but the answers are fixed.

### The Advanced Version
For this one, I took GPT-2 (that's a big language model) and trained it specifically on ALU conversations. It can generate new responses on the fly, so the answers feel more natural. But it needs a good computer to run and took way longer to set up.

### The Data
I used a conversational dataset from Hugging Face that contains university admissions dialogues. The dataset includes various conversation patterns about admissions, programs, costs, and requirements that I adapted specifically for ALU. It covers about 13 different types of questions students commonly ask.

## Testing the Chatbot

### How I Tested It
I made a list of questions that real students might ask (you can see them in test_questions.txt). I tested both versions of the bot to see which one gave better answers.

For the advanced GPT-2 version, I used some fancy metrics like BLEU score and perplexity to measure how good the answers are. You can run the evaluation yourself with:

```bash
cd src
python evaluate_model.py
```

### Example Conversations
Here are some test chats I tried:

**Student:** What programmes do you offer?
**Bot:** ALU offers three bachelor's degrees: Bachelor of Software Engineering (BSE) - combines technical skills with leadership...

**Student:** How much does ALU cost?
**Bot:** ALU lists tuition around US$3,000/year for Software Engineering and Entrepreneurial Leadership...

**Student:** I want to apply to ALU
**Bot:** Great! To apply to ALU: 1) Register on ALU's admission portal NOW at https://start.alueducation.com...

## What's In This Project

```
├── data/                        # Where I store the conversation data
│   ├── train_dataset_public.csv # Training conversations
│   ├── val_dataset_public.csv   # Validation data
│   └── test_dataset_public.csv  # Test conversations
├── models/                      # Saved trained models
│   └── data.pth                 # The neural network I trained
├── src/                         # All the Python code
│   ├── app.py                   # The web server that runs the chatbot
│   ├── chatbot_model.py         # Code for the advanced GPT-2 version
│   ├── evaluate_model.py        # Testing and evaluation stuff
│   ├── model.py                 # The neural network design
│   ├── nltk_utils.py            # Text processing helpers
│   ├── train.py                 # Script to train the simple model
│   ├── chat.py                  # Extra chat functions
│   └── intents.json             # Processed dataset from Hugging Face for training
├── web/                         # The website files
│   ├── index.html               # The main chat page
│   ├── styles.css               # Makes it look nice
│   ├── script.js                # Makes the chat work
│   └── test_questions.txt       # Questions I used for testing
├── ALU_Chatbot_Project.ipynb    # My Jupyter notebook with everything
├── LICENSE                      # License stuff
├── .gitignore                   # Git ignore file
└── README.md                   # This file you're reading
```

## How to Try It Out

### Quick Demo
1. First train the chatbot: `cd src && python train.py`
2. Start the web server: `cd src && python app.py`
3. Open http://localhost:8000 in your browser
4. Try asking questions from the test_questions.txt file
5. Watch the typing animation - I added that to make it feel more real

### Full Process
If you want to see everything:
1. Train either model (simple or advanced)
2. Run the evaluation to see how well it works
3. Start the web interface
4. Chat with it!

### Demo Video
I made a video showing how it all works. Check it out [here](https://youtu.be/LtUOlQm8Z3A?si=yvrCzfo09KEtj8IV) (link will be updated once uploaded).

## � Future Enhancements

- **Multi-turn Conversations**: Context awareness across dialogue turns
- **Larger Models**: Experiment with GPT-2 Medium/Large or other architectures
- **API Integration**: Connect with ALU's actual application portal
- **Multilingual Support**: Expand to French/Swahili for African students
- **Voice Interface**: Add speech-to-text capabilities

## 📄 References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [ALU Official Website](https://www.alueducation.com)
- [BLEU Score Paper](https://aclanthology.org/P02-1040.pdf)

## 👥 Team

**Student**: Jean Pierre NYONGABO
**Course**: Machine Learning technique I
**Institution**: African Leadership University
