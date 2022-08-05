#!/usr/bin/env python
# coding: utf-8

import requests
import numpy as np 

ride = {'SYMPTOM_COUGH': 0,
 'SYMPTOM_COLD':  0,
 'SYMPTOM_AIR_INSUFFICIENCY': 0,
 'SYMPTOM_FEVER': 0,
 'SYMPTOM_LOW_OXYGEN_SATURATION': 0,
 'SYMPTOM_BREATHING_CONDITION': 0,
 'SYMPTOM_TORACIC_APERTURE': 0,
 'SYMPTOM_THROAT_CONDITION': 0,
 'SYMPTOM_HEADACHE': 0,
 'SYMPTOM_BODY_PAIN': 0,
 'SYMPTOM_DIARRHEA':  0,
 'SYMPTOM_RUNNY_NOSE': 0,
 'SYMPTOM_NOSE_CONGESTION': 0,
 'SYMPTOM_WEAKNESS': 0,
 'SYMPTOM_ANOSMIA_OR_HYPOSMIA':  0,
 'SYMPTOM_NAUSEA': 0,
 'SYMPTOM_LACK_OF_APPETITE': 0,
 'SYMPTOM_ABDOMINAL_PAIN': 0,
 'SYMPTOM_CONSCIOUSNESS_DEGRADATION': 0,
 'age_group_0 - 5': 0,
 'age_group_16 - 25': 0,
 'age_group_26 - 40': 0,
 'age_group_41 - 60': 0,
 'age_group_6 - 15': 1,
 'age_group_61 - 80': 0,
 'age_group_>80': 0
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print()
print(response.json())


# python test_api.py