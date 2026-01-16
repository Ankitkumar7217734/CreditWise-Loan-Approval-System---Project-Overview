# CreditWise Loan System

## Problem Statement

A mid-sized financial company named **SecureTrust Bank** offers personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch applications.

Until now, SecureTrust Bank has been using a manual verification process where loan officers evaluate applications by checking income proofs, employment details, credit history, and other documents. This process is time-consuming, biased, and inconsistent.

As a result, the bank faces two major challenges:

1. Good customers sometimes get rejected, leading to loss of business.
2. High-risk customers sometimes get approved, leading to financial losses.

To solve this problem, the bank wants to introduce an intelligent loan approval system powered by Machine Learning that can automatically analyze applicant details and predict whether a loan should be **Approved** or **Rejected** before final human verification.

You are hired as a Machine Learning Engineer to design and develop this intelligent system using historical loan application data. The system must learn hidden patterns from previous customer records and provide accurate, fast, and unbiased loan approval decisions.

---

## Dataset Description

Each row in the dataset represents a loan applicant and contains multiple attributes describing their personal, financial, and credit information.

| Column Name | Description |
|------------|------------|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration in months |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| Loan_Approved (Target) | 1 = Approved, 0 = Rejected |
