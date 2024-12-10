import os
import json
import datetime
import csv
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from plyer import notification

# SSL context for nltk download
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Store tasks in a CSV file
TASK_FILE = "tasks.csv"

# Check if the task file exists, if not, create it
if not os.path.exists(TASK_FILE):
    with open(TASK_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Task', 'Deadline', 'Priority'])  # Header

# Function to add a task to CSV file
def add_task(task, deadline):
    priority = "Medium"  # Default priority
    task_data = [task, deadline, priority]
    with open(TASK_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(task_data)

# Function to retrieve tasks sorted by deadline
def get_next_task():
    tasks = []
    with open(TASK_FILE, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        tasks = sorted(list(reader), key=lambda x: x[1])  # Sort by deadline
    
    if tasks:
        next_task = tasks[0]  # First task is the next one to do
        task_name = next_task[0]
        deadline_str = next_task[1]
        deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M:%S")
        time_remaining = deadline - datetime.datetime.now()
        
        # Send notification if the task is due soon (within 5 minutes)
        if time_remaining.total_seconds() < 300:  # 5 minutes
            send_local_notification(task_name, str(time_remaining))
        
        return f"Your next task is: {task_name} with deadline {next_task[1]}"
    else:
        return "You have no upcoming tasks!"

# Function to send a local notification
def send_local_notification(task_name, time_remaining):
    notification.notify(
        title=f"Upcoming Task: {task_name}",
        message=f"You have {time_remaining} left to complete this task.",
        timeout=10  # Notification duration in seconds
    )

# Function to predict user input and return response
def chatbot(input_text):
    # Check if the input has the "by" keyword indicating a task with a deadline
    if " by " in input_text:  
        task_info = input_text.split(" by ")  # Split the task and deadline
        if len(task_info) == 2:
            task = task_info[0].strip()  # Task is everything before "by"
            deadline = task_info[1].strip()  # Deadline is everything after "by"
            add_task(task, deadline)  # Add the task to the task file
            return "Task added successfully!"  # Inform the user that the task was added

    # If no task information is found, process it for classification (using the vectorizer)
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]  # Predict the intent based on the input

    for intent in intents['intents']:
        if intent['tag'] == tag:
            if tag == "next_task":
                return get_next_task()  # Respond with the next task
            else:
                return random.choice(intent['responses'])  # Return a random response from the matched intent

    return "Sorry, I didn't understand that."  # Fallback response


def main():
    st.title("Task Management Chatbot")

    # Sidebar menu
    menu = ["Home", "Task History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot! Type your task, and I will manage it for you.")

        user_input = st.text_input("Enter task or ask for next task:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120)

    elif choice == "Task History":
        st.write("Here are your stored tasks:")
        with open(TASK_FILE, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                st.write(f"Task: {row[0]}, Deadline: {row[1]}, Priority: {row[2]}")

    elif choice == "About":
        st.write("""
        This chatbot helps you manage your tasks and prioritize them based on deadlines.
        You can input your tasks along with deadlines, and it will keep track of them.
        When you ask, it will tell you your next task to do.
        """)

if __name__ == "__main__":
    main()
