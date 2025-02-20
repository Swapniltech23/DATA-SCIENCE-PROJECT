## DATA SCIENCE PROJECT 

*COMAPNY : CODTECHIT SOLUTIONS

*NAME : SWAPNIL SAHU

*INTERN ID : CT08JDB

*DOMAIN : DATA SCIENCE

*DURATION : 4 WEEKS

*MENTOR : NEELA SANTOSH

#DESCRIPTION OF TASK :

### **End-to-End Fake News Detection Project: API, App, Real-World Applications & Future Scope**  

#### **Project Overview**  
This project focuses on **fake news detection**, an essential tool in today’s digital age where misinformation spreads rapidly. 
The project involves developing a **Machine Learning (ML) model**, deploying it as a **Flask API**, and integrating it with a **Streamlit web application** for user interaction.  

---

## **Development Process**  

### **1. Data Collection & Model Training**  
The first step was collecting and preprocessing a **dataset of news articles**. 
The data was cleaned, tokenized, and converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**. 
A **Logistic Regression** model was trained to classify articles as **Real or Fake**, achieving an accuracy of **91.55%**.  

---

### **2. Creating the Flask API**  
To make the model accessible, we developed a **Flask API** that accepts text input and returns a classification prediction.  
#### **Key Steps in API Development:**  
- Load the **trained model** and **vectorizer**.  
- Define an **API route** (`/predict`) that receives news text in JSON format.  
- Process the input, predict its authenticity, and return the result.  
- The API was tested using **Postman** and **cURL requests** in PowerShell.  

**Example API Response:**  
```json
{
  "confidence": 0.98,
  "prediction": "Fake"
}
```  

---

### **3. Developing the Streamlit App**  
To provide a user-friendly interface, we built a **Streamlit web app** that connects to the API.  
#### **Features of the Streamlit App:**  
- Users can input news articles for classification.  
- The app sends the text to the Flask API and displays the prediction.  
- Interactive and easy-to-use interface for non-technical users.  

**Command to Run the App:**  
```powershell
streamlit run app.py
```  

---

### **4. Deployment on GitHub**  
To make the project accessible, we **uploaded it to GitHub** for version control and easy sharing. The steps included:  
1. Creating a GitHub **repository**.  
2. **Initializing Git** and connecting the local project.  
3. **Committing and pushing** files to GitHub.  

**Commands Used:**  
```powershell
git init
git add .
git commit -m "Initial commit - Fake News Detection"
git remote add origin <GitHub-Repo-URL>
git push -u origin main
```  

---

## **Real-World Applications**  

### **1. Journalism & Media Verification**  
- News agencies can verify articles before publishing.  
- Prevents misinformation from spreading.  

### **2. Social Media Platforms**  
- Can be integrated into **Facebook, Twitter, and Instagram** to flag fake news.  

### **3. Corporate & Brand Protection**  
- Helps businesses detect **false reports** affecting their reputation.  

### **4. Government & Cybersecurity**  
- Used by agencies to track **misinformation campaigns and propaganda**.  

### **5. Education & Research**  
- Academic institutions can study **patterns in fake news**.  

---

## **Future Scope & Enhancements**  

### **1. Improved Model Performance**  
- Implement **Deep Learning models** like **BERT or GPT** for better accuracy.  
- Use **real-time learning** based on user feedback.  

### **2. Multilingual Support**  
- Expand detection capabilities to other **languages**.  

### **3. Cloud Deployment**  
- Deploy API on **AWS, Google Cloud, or Azure** for large-scale use.  

### **4. Integration with Fact-Checking Databases**  
- Cross-check articles with sources like **PolitiFact, Snopes, and Google Fact Check**.  

---

## **Conclusion**  
This project successfully **combines machine learning, API development, and web deployment** into a fully functional **Fake News Detection System**. 
By **deploying it via Flask API and Streamlit**, it ensures easy access for users. 
With the rise of misinformation, such AI-powered tools will become increasingly vital in maintaining **information integrity**. 
Future advancements can further enhance its accuracy, scalability, and real-time detection capabilities.


#output

![Image](https://github.com/user-attachments/assets/b79fce5f-5471-4c8b-95c9-bb05cf59e334)
