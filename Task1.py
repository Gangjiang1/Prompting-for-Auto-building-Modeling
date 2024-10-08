import numpy as np
import torch
import jsonlines 
import random 
import os

np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, TextStreamer, AutoConfig, AutoModelForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json

# load task1 test dataset
with open('Tesk1_test_dataset_example.json', 'r') as file:
    simple_data = json.load(file)

simple_val_input = simple_data[9]["Prompt"]
simple_val_target = simple_data[9]["Idf"]
simple_ex1_input = simple_data[0]["Prompt"]
simple_ex1_target = simple_data[0]["Idf"]
simple_ex2_input = simple_data[1]["Prompt"]
simple_ex2_target = simple_data[1]["Idf"]
simple_ex3_input = simple_data[2]["Prompt"]
simple_ex3_target = simple_data[2]["Idf"]

checkpoint = "<LLM checkpoint>" 
# e.g., 
# meta-llama/Meta-Llama-3-8B-Instruct
# Qwen/Qwen2-7B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# etc.

model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map='auto',cache_dir="<YOUR CACHE>")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

### zero-shot for task 1
simple_zero_shot = [
    {"role": "user", "content": f"""
You are an EnergyPlus IDF generator. Please output in IDF format:
Simulate a building that is 355.6 meters long, 137.3 meters wide, and 39.4 meters high. Only have one window on the east wall. The window sill height is 13.8 meters, the window height is 25.6 meters, and the window jamb width is 0.1 meters.
"""}
] 

### one-shot for task 1
simple_one_shot = [
    {"role": "user", "content": f"{simple_ex1_input}"},
    {"role": "assistant", "content":f"{simple_ex1_target}"},
    {"role": "user", "content": f"{simple_val_input}"}
] 

### two-shot for task 1
simple_two_shot = [
    {"role": "user", "content": f"{simple_ex1_input}"},
    {"role": "assistant", "content":f"{simple_ex1_target}"},
    {"role": "user", "content": f"{simple_ex2_input}"},
    {"role": "assistant", "content":f"{simple_ex2_target}"},
    {"role": "user", "content": f"{simple_val_input}"}
] 

### three-shot for task 1
simple_three_shot = [
    {"role": "user", "content": f"{simple_ex1_input}"},
    {"role": "assistant", "content":f"{simple_ex1_target}"},
    {"role": "user", "content": f"{simple_ex2_input}"},
    {"role": "assistant", "content":f"{simple_ex2_target}"},
    {"role": "user", "content": f"{simple_ex3_input}"},
    {"role": "assistant", "content":f"{simple_ex3_target}"},
    {"role": "user", "content": f"{simple_val_input}"}
] 

# explanation task 1
simple_reasoning = [
    {"role": "user", "content": "Simulate a building that is 86.2 meters long, 57.0 meters wide, and 16.0 meters high. The window sill height is 3.2 meters, the window height is 12.8 meters, and the window jamb width is 0.1 meters. Let's think step by step."},
    {"role": "assistant", "content": f"""
The building's long is 86.2 meters, wide is 57.0 meters and high is 16.0 meters.
Thus the floor:
Coordinate of floor face 0 x1=0.0,
Coordinate of floor face 0 y1=0.0,
Coordinate of floor face 0 z1=0.0,
Coordinate of floor face 0 x2=0.0,
Coordinate of floor face 0 y2=57.0,
Coordinate of floor face 0 z2=0.0,
Coordinate of floor face 0 x3=86.2,
Coordinate of floor face 0 y3=57.0,
Coordinate of floor face 0 z3=0.0,
Coordinate of floor face 0 x4=86.2,
Coordinate of floor face 0 y4=0.0,
Coordinate of floor face 0 z4=0.0;

The roof:
Coordinate of roof face 1 x1=86.2,
Coordinate of roof face 1 y1=0.0,
Coordinate of roof face 1 z1=16.0,
Coordinate of roof face 1 x2=86.2,
Coordinate of roof face 1 y2=57.0,
Coordinate of roof face 1 z2=16.0,
Coordinate of roof face 1 x3=0.0,
Coordinate of roof face 1 y3=57.0,
Coordinate of roof face 1 z3=16.0,
Coordinate of roof face 1 x4=0.0,
Coordinate of roof face 1 y4=0.0,
Coordinate of roof face 1 z4=16.0;

The wall:
Coordinate of wall face 2 x1=0.0,
Coordinate of wall face 2 y1=0.0,
Coordinate of wall face 2 z1=16.0,
Coordinate of wall face 2 x2=0.0,
Coordinate of wall face 2 y2=57.0,
Coordinate of wall face 2 z2=16.0,
Coordinate of wall face 2 x3=0.0,
Coordinate of wall face 2 y3=57.0,
Coordinate of wall face 2 z3=0.0,
Coordinate of wall face 2 x4=0.0,
Coordinate of wall face 2 y4=0.0,
Coordinate of wall face 2 z4=0.0;

Coordinate of wall face 4 x1=0.0,
Coordinate of wall face 4 y1=57.0,
Coordinate of wall face 4 z1=16.0,
Coordinate of wall face 4 x2=86.2,
Coordinate of wall face 4 y2=57.0,
Coordinate of wall face 4 z2=16.0,
Coordinate of wall face 4 x3=86.2,
Coordinate of wall face 4 y3=57.0,
Coordinate of wall face 4 z3=0.0,
Coordinate of wall face 4 x4=0.0,
Coordinate of wall face 4 y4=57.0,
Coordinate of wall face 4 z4=0.0;

Coordinate of wall face 6 x1=86.2,
Coordinate of wall face 6 y1=57.0,
Coordinate of wall face 6 z1=16.0,
Coordinate of wall face 6 x2=86.2,
Coordinate of wall face 6 y2=0.0,
Coordinate of wall face 6 z2=16.0,
Coordinate of wall face 6 x3=86.2,
Coordinate of wall face 6 y3=0.0,
Coordinate of wall face 6 z3=0.0,
Coordinate of wall face 6 x4=86.2,
Coordinate of wall face 6 y4=57.0,
Coordinate of wall face 6 z4=0.0;

Coordinate of wall face 8 x1=86.2,
Coordinate of wall face 8 y1=0.0,
Coordinate of wall face 8 z1=16.0,
Coordinate of wall face 8 x2=0.0,
Coordinate of wall face 8 y2=0.0,
Coordinate of wall face 8 z2=16.0,
Coordinate of wall face 8 x3=0.0,
Coordinate of wall face 8 y3=0.0,
Coordinate of wall face 8 z3=0.0,
Coordinate of wall face 8 x4=86.2,
Coordinate of wall face 8 y4=0.0,
Coordinate of wall face 8 z4=0.0;

The window sill height is 3.2 meters, the window height is 12.8 meters, and the window jamb width is 0.1 meters.

Thus the windows:
Coordinate of window face 3 x1=0.0,
Coordinate of window face 3 y1=56.9,
Coordinate of window face 3 z1=12.8,
Coordinate of window face 3 x2=0.0,
Coordinate of window face 3 y2=56.9,
Coordinate of window face 3 z2=3.2,
Coordinate of window face 3 x3=0.0,
Coordinate of window face 3 y3=0.1,
Coordinate of window face 3 z3=3.2,
Coordinate of window face 3 x4=0.0,
Coordinate of window face 3 y4=0.1,
Coordinate of window face 3 z4=12.8;

Therefore, the IDF code is:

BuildingSurface:Detailed,
    face 0,
    floor,
    extslabcarpet 4in climatezone 1-8,
    thermal zone 1,
    ,
    ground,
    ,
    nosun,
    nowind,
    ,
    ,
    0.0,
    0.0,
    0.0,
    0.0,
    57.0,
    0.0,
    86.2,
    57.0,
    0.0,
    86.2,
    0.0,
    0.0;

BuildingSurface:Detailed,
    face 1,
    roof,
    ashrae 189.1-2009 extroof iead climatezone 2-5,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    86.2,
    57.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    0.0,
    16.0;

BuildingSurface:Detailed,
    face 2,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    0.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    57.0,
    0.0,
    0.0,
    0.0,
    0.0;

BuildingSurface:Detailed,
    face 4,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    57.0,
    16.0,
    86.2,
    57.0,
    16.0,
    86.2,
    57.0,
    0.0,
    0.0,
    57.0,
    0.0;

BuildingSurface:Detailed,
    face 6,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    57.0,
    16.0,
    86.2,
    0.0,
    16.0,
    86.2,
    0.0,
    0.0,
    86.2,
    57.0,
    0.0;

BuildingSurface:Detailed,
    face 8,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    0.0,
    0.0,
    16.0,
    0.0,
    0.0,
    0.0,
    86.2,
    0.0,
    0.0;

FenestrationSurface:Detailed,
    face 3,
    window,
    ashrae 189.1-2009 extwindow climatezone 4-5,
    face 2,
    ,
    ,
    ,
    ,
    ,
    0.0,
    56.9,
    12.8,
    0.0,
    56.9,
    3.2,
    0.0,
    0.1,
    3.2,
    0.0,
    0.1,
    12.8;

    """
    },
    {"role": "user", "content": "Simulate a building that is 355.6 meters long, 137.3 meters wide, and 39.4 meters high. The window sill height is 13.8 meters, the window height is 25.6 meters, and the window jamb width is 0.1 meters. Let's think step by step."}
] 

# division task 1
simple_chatting = [
    {"role": "user", "content": f"""Simulate a building that is 86.2 meters long, 57.0 meters wide, and 16.0 meters high. The window sill height is 3.2 meters, the window height is 12.8 meters, and the window jamb width is 0.1 meters."""},
    {"role": "assistant", "content": f"""This building have one floor, one roof, four walls, and one window."""},
    {"role": "user", "content": f"""The floor's length is 86.2, width is 57.0, thus the details of floor is?"""},
    {"role": "assistant", "content": f"""
The building details of floor is:
BuildingSurface:Detailed,
    face 0,
    floor,
    extslabcarpet 4in climatezone 1-8,
    thermal zone 1,
    ,
    ground,
    ,
    nosun,
    nowind,
    ,
    ,
    0.0,
    0.0,
    0.0,
    0.0,
    57.0,
    0.0,
    86.2,
    57.0,
    0.0,
    86.2,
    0.0,
    0.0;"""},
    {"role": "user", "content": f"""The roof's length is 86.2, width is 57.0, height is 16.0, thus the details of roof is?"""},
    {"role": "assistant", "content": f"""
The building details of floor is:
BuildingSurface:Detailed,
    face 1,
    roof,
    ashrae 189.1-2009 extroof iead climatezone 2-5,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    86.2,
    57.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    0.0,
    16.0;"""},
    {"role": "user", "content": f"""The first wall's width is 57.0, height is 16.0, thus the details of first wall is?"""},
    {"role": "assistant", "content": f"""
The building details of first wall is:
BuildingSurface:Detailed,
    face 2,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    0.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    57.0,
    0.0,
    0.0,
    0.0,
    0.0;"""},
    {"role": "user", "content": f"""The second wall's length is 86.2, width is 57.0, height is 16.0, thus the details of second wall is?"""},
    {"role": "assistant", "content": f"""
The building details of second wall is:
BuildingSurface:Detailed,
    face 4,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    57.0,
    16.0,
    86.2,
    57.0,
    16.0,
    86.2,
    57.0,
    0.0,
    0.0,
    57.0,
    0.0;"""},
    {"role": "user", "content": f"""The third wall's length is 86.2, width is 57.0, height is 16.0, thus the details of third wall is?"""},
    {"role": "assistant", "content": f"""
The building details of third wall is:

BuildingSurface:Detailed,
    face 6,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    57.0,
    16.0,
    86.2,
    0.0,
    16.0,
    86.2,
    0.0,
    0.0,
    86.2,
    57.0,
    0.0;"""},
    {"role": "user", "content": f"""The fourth wall's length is 86.2, height is 16.0, thus the details of fourth wall is?"""},
    {"role": "assistant", "content": f"""
The building details of fourth wall is:
BuildingSurface:Detailed,
    face 8,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    0.0,
    0.0,
    16.0,
    0.0,
    0.0,
    0.0,
    86.2,
    0.0,
    0.0;"""},
    {"role": "user", "content": f"""The window sill height is 3.2 meters, the window height is 12.8 meters, and the window jamb width is 0.1 meters. Thus the details of window is?"""},
    {"role": "assistant", "content": f"""
The building details of window is:
FenestrationSurface:Detailed,
    face 3,
    window,
    ashrae 189.1-2009 extwindow climatezone 4-5,
    face 2,
    ,
    ,
    ,
    ,
    ,
    0.0,
    56.9,
    12.8,
    0.0,
    56.9,
    3.2,
    0.0,
    0.1,
    3.2,
    0.0,
    0.1,
    12.8;"""},
    {"role": "user", "content": f"""Based on the above provided building details, list all building details together."""},
    {"role": "assistant", "content": f"""
BuildingSurface:Detailed,
    face 0,
    floor,
    extslabcarpet 4in climatezone 1-8,
    thermal zone 1,
    ,
    ground,
    ,
    nosun,
    nowind,
    ,
    ,
    0.0,
    0.0,
    0.0,
    0.0,
    57.0,
    0.0,
    86.2,
    57.0,
    0.0,
    86.2,
    0.0,
    0.0;

BuildingSurface:Detailed,
    face 1,
    roof,
    ashrae 189.1-2009 extroof iead climatezone 2-5,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    86.2,
    57.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    0.0,
    16.0;

BuildingSurface:Detailed,
    face 2,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    0.0,
    16.0,
    0.0,
    57.0,
    16.0,
    0.0,
    57.0,
    0.0,
    0.0,
    0.0,
    0.0;

BuildingSurface:Detailed,
    face 4,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    0.0,
    57.0,
    16.0,
    86.2,
    57.0,
    16.0,
    86.2,
    57.0,
    0.0,
    0.0,
    57.0,
    0.0;

BuildingSurface:Detailed,
    face 6,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    57.0,
    16.0,
    86.2,
    0.0,
    16.0,
    86.2,
    0.0,
    0.0,
    86.2,
    57.0,
    0.0;

BuildingSurface:Detailed,
    face 8,
    wall,
    ashrae 189.1-2009 extwall mass climatezone 4,
    thermal zone 1,
    ,
    outdoors,
    ,
    sunexposed,
    windexposed,
    ,
    ,
    86.2,
    0.0,
    16.0,
    0.0,
    0.0,
    16.0,
    0.0,
    0.0,
    0.0,
    86.2,
    0.0,
    0.0;

FenestrationSurface:Detailed,
    face 3,
    window,
    ashrae 189.1-2009 extwindow climatezone 4-5,
    face 2,
    ,
    ,
    ,
    ,
    ,
    0.0,
    56.9,
    12.8,
    0.0,
    56.9,
    3.2,
    0.0,
    0.1,
    3.2,
    0.0,
    0.1,
    12.8;"""},
    {"role": "user", "content": f"""Based on the above structure, simulate a building that is 355.6 meters long, 137.3 meters wide, and 39.4 meters high. The window sill height is 13.8 meters, the window height is 25.6 meters, and the window jamb width is 0.1 meters."""}
]


### tokenizing
simple_zero_shot_prompt = tokenizer.apply_chat_template(simple_zero_shot, tokenize=False, add_generation_prompt=True)
simple_one_shot_prompt = tokenizer.apply_chat_template(simple_one_shot, tokenize=False, add_generation_prompt=True)
simple_two_shot_prompt = tokenizer.apply_chat_template(simple_two_shot, tokenize=False, add_generation_prompt=True)
simple_three_shot_prompt = tokenizer.apply_chat_template(simple_three_shot, tokenize=False, add_generation_prompt=True)

simple_reasoning_prompt = tokenizer.apply_chat_template(simple_reasoning, tokenize=False, add_generation_prompt=True)
simple_chatting_prompt = tokenizer.apply_chat_template(simple_chatting, tokenize=False, add_generation_prompt=True)

### inputs
simple_zero_shot_prompt_inputs = tokenizer(simple_zero_shot_prompt, return_tensors="pt").to(model.device)
simple_one_shot_prompt_inputs = tokenizer(simple_one_shot_prompt, return_tensors="pt").to(model.device)
simple_two_shot_prompt_inputs = tokenizer(simple_two_shot_prompt, return_tensors="pt").to(model.device)
simple_three_shot_prompt_inputs = tokenizer(simple_three_shot_prompt, return_tensors="pt").to(model.device)

simple_reasoning_prompt_inputs = tokenizer(simple_reasoning_prompt, return_tensors="pt").to(model.device)
simple_chatting_prompt_inputs = tokenizer(simple_chatting_prompt, return_tensors="pt").to(model.device)

# outputs = model.generate(**inputs, use_cache=True)
simple_zero_shot_prompt_outputs = model.generate(**simple_zero_shot_prompt_inputs, use_cache=True, max_length=8192)
simple_one_shot_prompt_outputs = model.generate(**simple_one_shot_prompt_inputs, use_cache=True, max_length=8192)
simple_two_shot_prompt_outputs = model.generate(**simple_two_shot_prompt_inputs, use_cache=True, max_length=8192)
simple_three_shot_prompt_outputs = model.generate(**simple_three_shot_prompt_inputs, use_cache=True, max_length=8192)

simple_reasoning_prompt_outputs = model.generate(**simple_reasoning_prompt_inputs, use_cache=True, max_length=8192)
simple_chatting_prompt_outputs = model.generate(**simple_chatting_prompt_inputs, use_cache=True, max_length=8192)

# output text
simple_zero_shot_prompt_output_text = tokenizer.decode(simple_zero_shot_prompt_outputs[0])
simple_one_shot_prompt_output_text = tokenizer.decode(simple_one_shot_prompt_outputs[0])
simple_two_shot_prompt_output_text = tokenizer.decode(simple_two_shot_prompt_outputs[0])
simple_three_shot_prompt_output_text = tokenizer.decode(simple_three_shot_prompt_outputs[0])

simple_reasoning_prompt_output_text = tokenizer.decode(simple_reasoning_prompt_outputs[0])
simple_chatting_prompt_output_text = tokenizer.decode(simple_chatting_prompt_outputs[0])

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_zero_shot_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_zero_shot_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_one_shot_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_one_shot_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_two_shot_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_two_shot_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_three_shot_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_three_shot_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_explanation_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_reasoning_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("task1_division_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(simple_chatting_prompt_output_text)
print("\n")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

