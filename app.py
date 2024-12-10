import os
import json
import datetime
import csv
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from plyer import notification
import schedule
import time
import threading

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier for the chatbot
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Task Management system
tasks = []

def add_task(task_name, deadline_str):
    deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d %H:%M:%S")
    tasks.append({"task": task_name, "deadline": deadline})
    tasks.sort(key=lambda x: x['deadline'])  # Sort tasks by deadline

def get_next_task():
    if tasks:
        next_task = tasks[0]  # The task with the earliest deadline
        task_name = next_task["task"]
        deadline_str = next_task["deadline"].strftime("%Y-%m-%d %H:%M:%S")
        return f"Your next task is: {task_name} at {deadline_str}"
    return "You have no tasks scheduled."

# Function for local notifications
def send_local_notification(task_name, time_remaining):
    notification.notify(
        title=f"Upcoming Task: {task_name}",
        message=f"You have {time_remaining} left to complete this task.",
        timeout=10  # Notification duration in seconds
    )

# Function to check for upcoming tasks and send notifications
def check_for_upcoming_tasks():
    while True:
        if tasks:
            next_task = tasks[0]
            time_remaining = next_task["deadline"] - datetime.datetime.now()
            if time_remaining.total_seconds() < 300:  # 5 minutes
                send_local_notification(next_task["task"], str(time_remaining))
        time.sleep(60)  # Check every minute

# Function to start checking tasks in a separate thread
def start_task_checking():
    task_thread = threading.Thread(target=check_for_upcoming_tasks, daemon=True)
    task_thread.start()

# Streamlit app
def main():
    st.title("AI-Based Time Management and Task Prioritization")

    # Create a sidebar menu with options
    menu = ["Home", "Add Task", "Chatbot", "Task History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Task Management and Prioritization system.")
        st.write("You can add tasks, chat with the AI chatbot, and see your task history.")

    elif choice == "Add Task":
        st.subheader("Add a New Task")
        task_name = st.text_input("Task Name:")
        deadline_str = st.text_input("Deadline (YYYY-MM-DD HH:MM:SS):")

        if st.button("Add Task"):
            add_task(task_name, deadline_str)
            st.write(f"Task '{task_name}' added successfully with a deadline of {deadline_str}.")

    elif choice == "Chatbot":
        st.subheader("Chat with the AI Chatbot")
        user_input = st.text_input("You:", key="chatbot_input")
        
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key="chatbot_response")

    elif choice == "Task History":
        st.subheader("Task History")
        if tasks:
            for task in tasks:
                st.text(f"Task: {task['task']}, Deadline: {task['deadline'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.write("You have no tasks scheduled.")

    elif choice == "About":
        st.subheader("About")
        st.write("""
        This project is an AI-based Time Management and Task Prioritization system. The chatbot is built using Natural Language Processing (NLP) techniques and Logistic Regression. 
        The task manager allows you to add tasks with deadlines, and it will notify you about upcoming tasks. 
        The system also prioritizes tasks based on their deadlines.
        """)

if __name__ == '__main__':
    # Start task checking in a separate thread
    start_task_checking()
    main()
