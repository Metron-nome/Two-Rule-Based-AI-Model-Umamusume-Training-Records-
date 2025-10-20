"""
ARTIFICAL INTELLIGENCE Assessment 1: Umamusume Trainer Calculation for Umamusumes
"""

#[AI ASSISTANCE NOTE: The following plotting logic and structure were created using a generative AI tool (Gemini AI).]
#[AI PROMPT USED] "Can you make a plot graph that suitable for the comparison of stats of Uma_Data_Records and the comparison of Rule 1 and Rule 2 AI Performance Test"

import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

"""
ACTION POINTS
> Based from the game Umamusume: Pretty Derby game released on mobile phone and on PC, 
Action Points determine the action of what the player does to train her Umamusume (Horse).

Action_Point = 1 (SPEED)
Action_Point = 2 (STAMINA)
Action_Point = 3 (POWER)
Action_Point = 4 (GUTS)
Action_Point = 5 (WIT)
Action_Point = 6 (REST)
"""

Uma_Data_Records = [
    # Run 1: Oguri Cap - Low Speed, High Stamina (Action Priority: Speed Training)
    {'Name': "Oguri Cap", 'Energy': 87, 'Failure_Chance': 7, "Speed": 252, "Stamina": 750, "Power": 327, "Guts": 141, "Wit": 124, "Action_Point": 1},

    # Run 2: Special Week - Low Power, High Speed (Action Priority: Power Training)
    {'Name': "Special Week", 'Energy': 98, 'Failure_Chance': 0, "Speed": 632, "Stamina": 345, "Power": 246, "Guts": 182, "Wit": 121, "Action_Point": 3},

    # Run 3: Silence Suzuka - Low Stamina, High Risk (Action Priority: Rest/Safety)
    {'Name': "Silence Suzuka", 'Energy': 41, 'Failure_Chance': 24, "Speed": 580, "Stamina": 345, "Power": 400, "Guts": 141, "Wit": 124, "Action_Point": 6},

    # Run 4: Agnes Tachyon - Critical Risk/Low Energy (Action Priority: Rest/Safety)
    {'Name': "Agnes Tachyon", 'Energy': 22, 'Failure_Chance': 76, "Speed": 892, "Stamina": 456, "Power": 602, "Guts": 172, "Wit": 221, "Action_Point": 6},

    # Run 5: Gold Ship - Low Power, High Stamina (Action Priority: Power Training)
    {'Name': "Gold Ship", 'Energy': 52, 'Failure_Chance': 3, "Speed": 326, "Stamina": 652, "Power": 212, "Guts": 154, "Wit": 176, "Action_Point": 3},

    # Run 6: Daiwa Scarlet - Low Stamina, High Wit (Action Priority: Stamina Training)
    {'Name': "Daiwa Scarlet", 'Energy': 95, 'Failure_Chance': 0, "Speed": 580, "Stamina": 290, "Power": 500, "Guts": 450, "Wit": 600, "Action_Point": 2},

    # Run 7: Calstone Light O - Low Wit (Action Priority: Wit Training)
    {'Name': "Calstone Light O", 'Energy': 75, 'Failure_Chance': 5, "Speed": 450, "Stamina": 450, "Power": 450, "Guts": 350, "Wit": 200, "Action_Point": 5},

    # Run 8: Fuji Kiseki - Balanced but low Guts (Action Priority: Power/Guts)
    {'Name': "Fuji Kiseki", 'Energy': 65, 'Failure_Chance': 10, "Speed": 480, "Stamina": 480, "Power": 300, "Guts": 250, "Wit": 400, "Action_Point": 3},

    # Run 9: Tokai Teio - Low Speed (Action Priority: Speed Training)
    {'Name': "Tokai Teio", 'Energy': 88, 'Failure_Chance': 0, "Speed": 350, "Stamina": 500, "Power": 400, "Guts": 400, "Wit": 500, "Action_Point": 1},

    # Run 10: Tamamo Cross - Balanced/Average (Action Priority: Stamina Training)
    {'Name': "Tamamo Cross", 'Energy': 60, 'Failure_Chance': 15, "Speed": 420, "Stamina": 390, "Power": 400, "Guts": 400, "Wit": 400, "Action_Point": 2},
    
    # Run 11: Mejiro McQueen - Low Guts (Action Priority: Guts)
    {'Name': "Mejiro McQueen", 'Energy': 60, 'Failure_Chance': 5, "Speed": 450, "Stamina": 550, "Power": 390, "Guts": 290, "Wit": 480, "Action_Point": 4},
    
    # Run 12: Narita Brian - Critical Low Energy (Action Priority: Rest) - Tests Alg 1 Safety Override
    {'Name': "Narita Brian", 'Energy': 32, 'Failure_Chance': 10, "Speed": 510, "Stamina": 530, "Power": 510, "Guts": 400, "Wit": 500, "Action_Point": 6},
    
    # Run 13: King Halo - Very Low Wit (Action Priority: Wit)
    {'Name': "King Halo", 'Energy': 78, 'Failure_Chance': 5, "Speed": 590, "Stamina": 400, "Power": 410, "Guts": 410, "Wit": 150, "Action_Point": 5},
    
    # Run 14: Symboli Rudolf - Low Power (Action Priority: Power)
    {'Name': "Symboli Rudolf", 'Energy': 85, 'Failure_Chance': 0, "Speed": 500, "Stamina": 500, "Power": 320, "Guts": 450, "Wit": 500, "Action_Point": 3},
    
    # Run 15: Rice Shower - Low Stamina (Action Priority: Stamina)
    {'Name': "Rice Shower", 'Energy': 70, 'Failure_Chance': 10, "Speed": 400, "Stamina": 350, "Power": 400, "Guts": 400, "Wit": 400, "Action_Point": 2},
    
    # Run 16: Seiun Sky - High Failure Risk (Action Priority: Rest) - Tests Rule 1/Safety
    {'Name': "Seiun Sky", 'Energy': 45, 'Failure_Chance': 32, "Speed": 380, "Stamina": 520, "Power": 450, "Guts": 450, "Wit": 500, "Action_Point": 6},
    
    # Run 17: T.M. Opera O - Low Guts, High Energy (Action Priority: Guts)
    {'Name': "T.M. Opera O", 'Energy': 90, 'Failure_Chance': 0, "Speed": 550, "Stamina": 550, "Power": 550, "Guts": 200, "Wit": 500, "Action_Point": 4},

    # Run 18: Nice Nature - Very low Power (Action Priority: Power)
    {'Name': "Nice Nature", 'Energy': 65, 'Failure_Chance': 21, "Speed": 450, "Stamina": 450, "Power": 200, "Guts": 350, "Wit": 400, "Action_Point": 3},
    
    # Run 19: Marvelous Sunday - Low Speed, High Wit (Action Priority: Speed)
    {'Name': "Marvelous Sunday", 'Energy': 99, 'Failure_Chance': 0, "Speed": 330, "Stamina": 480, "Power": 480, "Guts": 450, "Wit": 550, "Action_Point": 1},
    
    # Run 20: Manhattan Cafe - Low Stamina, High Power, High Risk (Action Priority: Rest)
    {'Name': "Manhattan Cafe", 'Energy': 55, 'Failure_Chance': 75, "Speed": 400, "Stamina": 250, "Power": 580, "Guts": 400, "Wit": 500, "Action_Point": 6}
    
]

#The Umamusume Dataframe
UmaDataFrame = pd.DataFrame(Uma_Data_Records)

Stats_Input_For_Umas = UmaDataFrame.drop(['Action_Point', 'Name'], axis = 1)
Action_Input_Code = UmaDataFrame ['Action_Point']


"""
1ST RULE: Checking the the stats of the Uma (Heuristic Sequential Model)
It checks rules based on the order.

It checks if for example:

If an Uma's Energy is Lower than 40, and the Failure Chance is higher than 25,
Then the trainer should focus on taking the Action of choosing Rest than training.

If an Uma's Speed is lower than 400 and Energy is higher than 70
Then the trainer must priortize training the Uma's speed.
"""

def Checking_Stats_Of_Uma(Stats):

    if Stats['Energy'] < 40 or Stats['Failure_Chance'] >= 25:
        return 6 

    elif Stats['Speed'] < 400 and Stats['Energy'] >= 70:
        return 1
    
    elif Stats['Stamina'] < 400 and Stats['Speed'] >= 450:
        return 2 
    
    elif Stats['Guts'] < 400 and Stats['Power'] < 400:
        return 4 
        
    elif Stats['Wit'] < 500:
        return 5
        
    else:
        return 6

#This allows the 1st Rule to be applied to check the data for the Umas' Stats.
Trainer_Prediction = Stats_Input_For_Umas.apply(Checking_Stats_Of_Uma, axis=1)


"""
RULE 2: Uma Scoring System (Weighted Scoring)

The 2nd Rule weights based on the training priority of the Umas.
"""
Stat_Points = {
    "Training_Risk_Penalty": 25,
    "Speed_Priority": 10,
    "Stamina_Priority": 9,
    "Power_Priority": 7,
    "Guts_Priorty": 8,
    "Wit_Priority": 5,
}

def Uma_Weighted_Score(Stats):
    
    Uma_Score = 0
    
    if Stats['Failure_Chance'] >= 15 or Stats['Energy'] < 40:
        Uma_Score -= Stat_Points['Training_Risk_Penalty'] * 2 

    if Stats['Speed'] < 450:
        Uma_Score += Stat_Points['Speed_Priority']
    
    if Stats['Stamina'] < 450:
        Uma_Score += Stat_Points['Stamina_Priority']

    if Stats['Power'] < 450:
        Uma_Score += Stat_Points['Power_Priority']
    
    if Stats['Guts'] < 400:
        Uma_Score += Stat_Points['Guts_Priorty']
        
    if Stats['Wit'] < 500:
        Uma_Score += Stat_Points['Wit_Priority']

    if Uma_Score <= 0:
        return 6
    
  
    else:
        if Stats['Speed'] < 450:
             return 1  # Speed Training
        elif Stats['Stamina'] < 450:
             return 2  # Stamina Training
        elif Stats['Power'] < 450:
             return 3  # Power Training
        elif Stats['Guts'] < 400:
             return 4  # Guts Training
        else:
             return 5 # Wit Training

#This will apply the 2nd Rule for the Dataframe.
Trainer_Prediction_Weighted = Stats_Input_For_Umas.apply(Uma_Weighted_Score, axis=1)


#AI Performance Test

def AI_Performance_Test(Trainer_Action_True , Trainer_Prediction, Name):

    AI_Accuracy = accuracy_score(Trainer_Action_True, Trainer_Prediction)
    AI_Precision = precision_score(Trainer_Action_True, Trainer_Prediction, average = 'macro', zero_division = 0)
    AI_Recall = recall_score(Trainer_Action_True, Trainer_Prediction, average='macro', zero_division = 0)
    AI_F1_Score = f1_score(Trainer_Action_True, Trainer_Prediction, average = 'macro', zero_division = 0)
    
    print(f"\n Results for {Name}")
    print(f"Accuracy: {AI_Accuracy:.4f}")
    print(f"Precision (Macro): {AI_Precision:.4f}")
    print(f"Recall (Macro): {AI_Recall:.4f}")
    AI_F1_Score_Value = AI_F1_Score
    print(f"F1-Score (Macro): {AI_F1_Score_Value:.4f}")


    return {
        'Accuracy': AI_Accuracy,
        'Precision': AI_Precision,
        'Recall': AI_Recall,
        'F1-Score': AI_F1_Score
    }


Umamusume_Plot_Data = UmaDataFrame.set_index('Name')[["Speed", "Stamina", "Power", "Guts", "Wit"]]


plt.figure(figsize=(14, 7))
Umamusume_Plot_Data.plot(kind='bar', rot=45, ax=plt.gca())
plt.title('Umamusume Stat Comparison')
plt.ylabel('Stat Value')
plt.xlabel('Umamusume Name')
plt.legend(title='Stat Type', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


Rule_1_Result = AI_Performance_Test(Action_Input_Code, Trainer_Prediction, "1ST RULE: Checking the the stats of the Uma (Heuristic Sequential Model)")
Rule_2_Result = AI_Performance_Test(Action_Input_Code, Trainer_Prediction_Weighted, "2ND RULE: Uma Scoring System (Weighted Scoring)")


AI_Performance_Metric = pd.DataFrame({
    'Rule 1': Rule_1_Result,
    'Rule 2': Rule_2_Result,
})


plt.figure(figsize=(10, 6))
AI_Performance_Metric.plot(kind='bar', rot=0, ax=plt.gca())
plt.title('Performance Comparison of Rule-Based AI Algorithms')
plt.ylabel('Score (Macro Average)')
plt.xlabel('Metric')
plt.legend(title='Algorithm')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()