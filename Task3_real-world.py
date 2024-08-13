import numpy as np
import torch
import jsonlines 
import random 
import os

np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)

import torch
from transformers import AutoTokenizer, TextStreamer, AutoConfig, AutoModelForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = "<LLM checkpoint>" 
# e.g., 
# meta-llama/Meta-Llama-3-8B-Instruct
# Qwen/Qwen2-7B-Instruct
# mistralai/Mistral-7B-Instruct-v0.3
# etc.

model = AutoModelForCausalLM.from_pretrained(checkpoint,device_map='auto',cache_dir="<YOUR CACHE>")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#============================================================================================
### 1 case
#============================================================================================
real_one_shot = [
    {"role": "user", "content": """
Simulate a building that is 20 meters long, 10 meters wide, and 3 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.
The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.

The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.

The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.

The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;

Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;

Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;

Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;

Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;

Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;

Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;

Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;

Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;

Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;

Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;

WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;

Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;

Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;

Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;

Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;

Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;

People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;

ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;

ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Based on the above generated structure, simulate a building that is 8.23 meters long, 4.27 meters wide, and 3.35 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (0.765,3.505) meters.
The window's U-Factor is 1.266253, and its solar heat gain coefficient is 0.43.

The building accommodates 3 people with an activity level of 132, and the occupancy rates are as follows:
66.6% from Monday to Thursday during 8:00 to 17:00, 100% on Friday from 8:00 to 16:00, 0% on Saturday and Sunday.

The definition for lighting is 6 W/m2, and for electrical equipment, it is 4 W/m2. The infiltration rate is 1.4 ACH.

The heating setpoint is 22.2.0 degrees Celsius, and the cooling setpoint is 24.4 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exterior Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
exterior Finish OSB: 0.02 meters thick, with a conductivity of 0.22 W/m-K.
Rigid Insulation: 0.03 meters thick, with a conductivity of 0.05 W/m-K.
OSB: 0.01 meters thick, with a conductivity of 0.18 W/m-K.
Stud Layer: 0.15 meters thick, with a conductivity of 0.048 W/m-K.
EPDM Rubber: 0.0028 meters thick, with a conductivity of 0.25 W/m-K.
Insulted Truss: 0.90 meters thick, with a conductivity of 0.05 W/m-K.
Drywall: 0.02 meters thick, with a conductivity of 0.18 W/m-K.
Roddent Barrier: 0.002 meters thick, with a conductivity of 0.5 W/m-K.
Insulated Joist: 0.4 meters thick, with a conductivity of 0.05 W/m-K.
Plywood: 0.008 meters thick, with a conductivity of 0.22 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""}
] 

### one-shot
real_one_shot_prompt = tokenizer.apply_chat_template(real_one_shot, tokenize=False, add_generation_prompt=True)
real_one_shot_prompt_inputs = tokenizer(real_one_shot_prompt, return_tensors="pt").to(model.device)
real_one_shot_prompt_outputs = model.generate(**real_one_shot_prompt_inputs, use_cache=True, max_length=8000)
real_one_shot_prompt_output_text = tokenizer.decode(real_one_shot_prompt_outputs[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("real_one_shot_prompt_output_text")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(real_one_shot_prompt_output_text)

#============================================================================================
### 2 cases
#============================================================================================
real_two_shot = [
    {"role": "user", "content": """
Simulate a building that is 20 meters long, 10 meters wide, and 3 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.
The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.

The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.

The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.

The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;

Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;

Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;

Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;

Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;

Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;

Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;

Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;

Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;

Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;

Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;

WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;

Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;

Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;

Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;

Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;

Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;

People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;

ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;

ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;

"""},
    {"role": "user", "content": """
Simulate a building that is 40 meters long, 20 meters wide, and 6 meters high.

There is a door on the south-west side. The door is 1.82 meters wide and 4.44 meters high, with its bottom-left corner positioned at (0,1.82) meters from the south-west corner of the building.

There is a window on the east side. The window is 5.48 meters wide and 3.0 meters high. The window sill is 1.82 meters high, and the jamb position is (7.26,12.74) meters.
The window's U-Factor is 1.8, and its solar heat gain coefficient is 0.4.

The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
50.5% from Monday to Thursday during 8:00 to 17:00, 75% on Friday from 8:00 to 16:00, 30% on Saturday and Sunday.

The definition for lighting is 5.5 W/m2, and for electrical equipment, it is 7.5 W/m2. The infiltration rate is 1.8 ACH.

The heating setpoint is 21.8 degrees Celsius, and the cooling setpoint is 25.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 20, 0,
  40, 20, 0,
  40, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 0, 6,
  40, 20, 6,
  0, 20, 6,
  0, 0, 6;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 6,
  0, 20, 6,
  0, 20, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 1.82, 4.44,
  0, 1.82, 0,
  0, 0, 0,
  0, 0, 4.44;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 20, 6,
  40, 20, 6,
  40, 20, 0,
  0, 20, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 20, 6,
  40, 0, 6,
  40, 0, 0,
  40, 20, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  40, 7.26, 4.82,
  40, 7.26, 1.82,
  40, 12.74, 1.82,
  40, 12.74, 4.82;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 0, 6,
  0, 0, 6,
  0, 0, 0,
  40, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,21.8;

Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,25.0;

Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;

Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;

Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;

Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;

Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;

Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;

Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;

Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;

Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;

Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;

WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.8,0.4,0.42;

Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;

Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.75,24:00,0;

Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.505,24:00,0;

Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.3;

Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,5.5,,,,,1,General;

People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;

ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,7.5,,,,,General;

ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,1.8,0.03,0.003,0,;

"""},
    {"role": "user", "content": """
Based on the above generated structure, simulate a building that is 8.23 meters long, 4.27 meters wide, and 3.35 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (0.765,3.505) meters.
The window's U-Factor is 1.266253, and its solar heat gain coefficient is 0.43.

The building accommodates 3 people with an activity level of 132, and the occupancy rates are as follows:
66.6% from Monday to Thursday during 8:00 to 17:00, 100% on Friday from 8:00 to 16:00, 0% on Saturday and Sunday.

The definition for lighting is 6 W/m2, and for electrical equipment, it is 4 W/m2. The infiltration rate is 1.4 ACH.

The heating setpoint is 22.2.0 degrees Celsius, and the cooling setpoint is 24.4 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exterior Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
exterior Finish OSB: 0.02 meters thick, with a conductivity of 0.22 W/m-K.
Rigid Insulation: 0.03 meters thick, with a conductivity of 0.05 W/m-K.
OSB: 0.01 meters thick, with a conductivity of 0.18 W/m-K.
Stud Layer: 0.15 meters thick, with a conductivity of 0.048 W/m-K.
EPDM Rubber: 0.0028 meters thick, with a conductivity of 0.25 W/m-K.
Insulted Truss: 0.90 meters thick, with a conductivity of 0.05 W/m-K.
Drywall: 0.02 meters thick, with a conductivity of 0.18 W/m-K.
Roddent Barrier: 0.002 meters thick, with a conductivity of 0.5 W/m-K.
Insulated Joist: 0.4 meters thick, with a conductivity of 0.05 W/m-K.
Plywood: 0.008 meters thick, with a conductivity of 0.22 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""}
] 

### two-shot
real_two_shot_prompt = tokenizer.apply_chat_template(real_two_shot, tokenize=False, add_generation_prompt=True)
real_two_shot_prompt_inputs = tokenizer(real_two_shot_prompt, return_tensors="pt").to(model.device)
real_two_shot_prompt_outputs = model.generate(**real_two_shot_prompt_inputs, use_cache=True, max_length=12000)
real_two_shot_prompt_output_text = tokenizer.decode(real_two_shot_prompt_outputs[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("real_two_shot")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(real_two_shot_prompt_output_text)

#============================================================================================
### 3 cases
#============================================================================================
real_three_shot = [
    {"role": "user", "content": """
Simulate a building that is 20 meters long, 10 meters wide, and 3 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.
The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.

The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.

The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.

The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Simulate a building that is 40 meters long, 20 meters wide, and 6 meters high.

There is a door on the south-west side. The door is 1.82 meters wide and 4.44 meters high, with its bottom-left corner positioned at (0,1.82) meters from the south-west corner of the building.

There is a window on the east side. The window is 5.48 meters wide and 3.0 meters high. The window sill is 1.82 meters high, and the jamb position is (7.26,12.74) meters.
The window's U-Factor is 1.8, and its solar heat gain coefficient is 0.4.

The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
50.5% from Monday to Thursday during 8:00 to 17:00, 75% on Friday from 8:00 to 16:00, 30% on Saturday and Sunday.

The definition for lighting is 5.5 W/m2, and for electrical equipment, it is 7.5 W/m2. The infiltration rate is 1.8 ACH.

The heating setpoint is 21.8 degrees Celsius, and the cooling setpoint is 25.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 20, 0,
  40, 20, 0,
  40, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 0, 6,
  40, 20, 6,
  0, 20, 6,
  0, 0, 6;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 6,
  0, 20, 6,
  0, 20, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 1.82, 4.44,
  0, 1.82, 0,
  0, 0, 0,
  0, 0, 4.44;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 20, 6,
  40, 20, 6,
  40, 20, 0,
  0, 20, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 20, 6,
  40, 0, 6,
  40, 0, 0,
  40, 20, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  40, 7.26, 4.82,
  40, 7.26, 1.82,
  40, 12.74, 1.82,
  40, 12.74, 4.82;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  40, 0, 6,
  0, 0, 6,
  0, 0, 0,
  40, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,21.8;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,25.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.8,0.4,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.75,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.505,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.3;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,5.5,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,7.5,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,1.8,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Simulate a building that is 80 meters long, 40 meters wide, and 12 meters high.

There is a door on the south-west side. The door is 3.64 meters wide and 8.88 meters high, with its bottom-left corner positioned at (0,3.64) meters from the south-west corner of the building.

There is a window on the east side. The window is 10.96 meters wide and 6.0 meters high. The window sill is 3.64 meters high, and the jamb position is (14.52,25.48) meters.
The window's U-Factor is 0.8, and its solar heat gain coefficient is 0.2.

The building accommodates 8 people with an activity level of 128, and the occupancy rates are as follows:
80% from Monday to Thursday during 8:00 to 17:00, 90% on Friday from 8:00 to 16:00, 10% on Saturday and Sunday.

The definition for lighting is 2.8 W/m2, and for electrical equipment, it is 4.9 W/m2. The infiltration rate is 1.8 ACH.

The heating setpoint is 21.0 degrees Celsius, and the cooling setpoint is 24.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 40, 0,
  80, 40, 0,
  80, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  80, 0, 12,
  80, 40, 12,
  0, 40, 12,
  0, 0, 12;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 12,
  0, 40, 12,
  0, 40, 0,
  0, 0, 0;

FenestrationSurface:Detailed,
  Face 3,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 3.64, 8.88,
  0, 3.64, 0,
  0, 0, 0,
  0, 0, 8.88;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 40, 12,
  80, 40, 12,
  80, 40, 0,
  0, 40, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  80, 40, 12,
  80, 0, 12,
  80, 0, 0,
  80, 40, 0;

FenestrationSurface:Detailed,
  Face 6,
  Window,
  Window,
  Face 5,
  ,
  ,
  ,
  ,
  ,
  80, 14.52, 9.64,
  80, 14.52, 3.64,
  80, 25.48, 3.64,
  80, 25.48, 9.64;

BuildingSurface:Detailed,
  Face 7,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  80, 0, 12,
  0, 0, 12,
  0, 0, 0,
  80, 0, 0;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,21.0;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,24.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,0.8,0.2,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,128;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.9,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.8,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.1;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,2.8,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,8,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,4.9,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,1.8,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Based on the above generated structure, simulate a building that is 8.23 meters long, 4.27 meters wide, and 3.35 meters high.

There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.

There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (0.765,3.505) meters.
The window's U-Factor is 1.266253, and its solar heat gain coefficient is 0.43.

The building accommodates 3 people with an activity level of 132, and the occupancy rates are as follows:
66.6% from Monday to Thursday during 8:00 to 17:00, 100% on Friday from 8:00 to 16:00, 0% on Saturday and Sunday.

The definition for lighting is 6 W/m2, and for electrical equipment, it is 4 W/m2. The infiltration rate is 1.4 ACH.

The heating setpoint is 22.2.0 degrees Celsius, and the cooling setpoint is 24.4 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exterior Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
exterior Finish OSB: 0.02 meters thick, with a conductivity of 0.22 W/m-K.
Rigid Insulation: 0.03 meters thick, with a conductivity of 0.05 W/m-K.
OSB: 0.01 meters thick, with a conductivity of 0.18 W/m-K.
Stud Layer: 0.15 meters thick, with a conductivity of 0.048 W/m-K.
EPDM Rubber: 0.0028 meters thick, with a conductivity of 0.25 W/m-K.
Insulted Truss: 0.90 meters thick, with a conductivity of 0.05 W/m-K.
Drywall: 0.02 meters thick, with a conductivity of 0.18 W/m-K.
Roddent Barrier: 0.002 meters thick, with a conductivity of 0.5 W/m-K.
Insulated Joist: 0.4 meters thick, with a conductivity of 0.05 W/m-K.
Plywood: 0.008 meters thick, with a conductivity of 0.22 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""}
] 


### three-shot
real_three_shot_prompt = tokenizer.apply_chat_template(real_three_shot, tokenize=False, add_generation_prompt=True)
real_three_shot_prompt_inputs = tokenizer(real_three_shot_prompt, return_tensors="pt").to(model.device)
real_three_shot_prompt_outputs = model.generate(**real_three_shot_prompt_inputs, use_cache=True, max_length=16000)
real_three_shot_prompt_output_text = tokenizer.decode(real_three_shot_prompt_outputs[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("real_three_shot")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(real_three_shot_prompt_output_text)

#============================================================================================
### explanation
#============================================================================================

real_chat_shot = [
    {"role": "user", "content": """
Simulate a building that is 20 meters long, 10 meters wide, and 3 meters high.
There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.
There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.

The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.
The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.
The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.
The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
This building has one floor, one roof, four walls, one window, and one door. This building has various of building settings materials.
"""},
    {"role": "user", "content": """
This building is 20 meters long, 10 meters wide, and 3 meters high, thus the details of building geometry is?
"""},
    {"role": "assistant", "content": """
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

BuildingSurface:Detailed,
  Face 3,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;
"""},
    {"role": "user", "content": """
The door is 0.91 meters wide and 2.22 meters high. The door position is (0,0.91) meters. The details of door is?
"""},
    {"role": "assistant", "content": """
FenestrationSurface:Detailed,
  Face 7,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;
"""},
    {"role": "user", "content": """
The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters. The details of window is?
"""},
    {"role": "assistant", "content": """
FenestrationSurface:Detailed,
  Face 8,
  Window,
  Window,
  Face 4,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;
"""},
    {"role": "user", "content": """
The building settings and building materials are below:

The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.
The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.
The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.
The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.
For materials

Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;
"""},
    {"role": "user", "content": f"""Based on the above provided building details, list all building details together."""},
    {"role": "assistant", "content": f"""
BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

BuildingSurface:Detailed,
  Face 3,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;

FenestrationSurface:Detailed,
  Face 7,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;

FenestrationSurface:Detailed,
  Face 8,
  Window,
  Window,
  Face 4,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Based on the above generated structure, simulate a building that is 8.23 meters long, 4.27 meters wide, and 3.35 meters high.
There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.
There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (0.765,3.505) meters.

The window's U-Factor is 1.266253, and its solar heat gain coefficient is 0.43.
The building accommodates 3 people with an activity level of 132, and the occupancy rates are as follows:
66.6% from Monday to Thursday during 8:00 to 17:00, 100% on Friday from 8:00 to 16:00, 0% on Saturday and Sunday.
The definition for lighting is 6 W/m2, and for electrical equipment, it is 4 W/m2. The infiltration rate is 1.4 ACH.
The heating setpoint is 22.2.0 degrees Celsius, and the cooling setpoint is 24.4 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exterior Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
exterior Finish OSB: 0.02 meters thick, with a conductivity of 0.22 W/m-K.
Rigid Insulation: 0.03 meters thick, with a conductivity of 0.05 W/m-K.
OSB: 0.01 meters thick, with a conductivity of 0.18 W/m-K.
Stud Layer: 0.15 meters thick, with a conductivity of 0.048 W/m-K.
EPDM Rubber: 0.0028 meters thick, with a conductivity of 0.25 W/m-K.
Insulted Truss: 0.90 meters thick, with a conductivity of 0.05 W/m-K.
Drywall: 0.02 meters thick, with a conductivity of 0.18 W/m-K.
Roddent Barrier: 0.002 meters thick, with a conductivity of 0.5 W/m-K.
Insulated Joist: 0.4 meters thick, with a conductivity of 0.05 W/m-K.
Plywood: 0.008 meters thick, with a conductivity of 0.22 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""}
] 

### explanation
real_chat_shot_prompt = tokenizer.apply_chat_template(real_chat_shot, tokenize=False, add_generation_prompt=True)
real_chat_shot_prompt_inputs = tokenizer(real_chat_shot_prompt, return_tensors="pt").to(model.device)
real_chat_shot_prompt_outputs = model.generate(**real_chat_shot_prompt_inputs, use_cache=True, max_length=16000)
real_chat_shot_prompt_output_text = tokenizer.decode(real_chat_shot_prompt_outputs[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("real_explanation")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(real_chat_shot_prompt_output_text)

#============================================================================================
### division
#============================================================================================

real_reason_shot = [
    {"role": "user", "content": """
Simulate a building that is 20 meters long, 10 meters wide, and 3 meters high.
There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.
There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.

The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.
The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.
The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.
The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.
"""},
    {"role": "assistant", "content": """
This building has one floor, one roof, four walls, one window, and one door. This building has various of building settings materials.

The building is 20 meters long, 10 meters wide, and 3 meters high.
For the floor,
Coordinate of floor face 0 x1=0,
Coordinate of floor face 0 y1=0,
Coordinate of floor face 0 z1=0,
Coordinate of floor face 0 x2=0,
Coordinate of floor face 0 y2=10,
Coordinate of floor face 0 z2=0,
Coordinate of floor face 0 x3=20,
Coordinate of floor face 0 y3=10,
Coordinate of floor face 0 z3=0,
Coordinate of floor face 0 x4=20,
Coordinate of floor face 0 y4=0,
Coordinate of floor face 0 z4=0;

For the roof,
Coordinate of roof face 1 x1=20,
Coordinate of roof face 1 y1=0,
Coordinate of roof face 1 z1=3,
Coordinate of roof face 1 x2=20,
Coordinate of roof face 1 y2=10,
Coordinate of roof face 1 z2=3,
Coordinate of roof face 1 x3=0,
Coordinate of roof face 1 y3=10,
Coordinate of roof face 1 z3=3,
Coordinate of roof face 1 x4=0,
Coordinate of roof face 1 y4=0,
Coordinate of roof face 1 z4=3;

For the wall,
Coordinate of wall face 2 x1=0,
Coordinate of wall face 2 y1=0,
Coordinate of wall face 2 z1=3,
Coordinate of wall face 2 x2=0,
Coordinate of wall face 2 y2=10,
Coordinate of wall face 2 z2=3,
Coordinate of wall face 2 x3=0,
Coordinate of wall face 2 y3=10,
Coordinate of wall face 2 z3=0,
Coordinate of wall face 2 x4=0,
Coordinate of wall face 2 y4=0,
Coordinate of wall face 2 z4=0;

Coordinate of wall face 3 x1=0,
Coordinate of wall face 3 y1=10,
Coordinate of wall face 3 z1=3,
Coordinate of wall face 3 x2=20,
Coordinate of wall face 3 y2=10,
Coordinate of wall face 3 z2=3,
Coordinate of wall face 3 x3=20,
Coordinate of wall face 3 y3=10,
Coordinate of wall face 3 z3=0,
Coordinate of wall face 3 x4=0,
Coordinate of wall face 3 y4=10,
Coordinate of wall face 3 z4=0;

Coordinate of wall face 4 x1=20,
Coordinate of wall face 4 y1=10,
Coordinate of wall face 4 z1=3,
Coordinate of wall face 4 x2=20,
Coordinate of wall face 4 y2=0,
Coordinate of wall face 4 z2=3,
Coordinate of wall face 4 x3=20,
Coordinate of wall face 4 y3=0,
Coordinate of wall face 4 z3=0,
Coordinate of wall face 4 x4=20,
Coordinate of wall face 4 y4=10,
Coordinate of wall face 4 z4=0;

Coordinate of wall face 5 x1=20,
Coordinate of wall face 5 y1=0,
Coordinate of wall face 5 z1=3,
Coordinate of wall face 5 x2=0,
Coordinate of wall face 5 y2=0,
Coordinate of wall face 5 z2=3,
Coordinate of wall face 5 x3=0,
Coordinate of wall face 5 y3=0,
Coordinate of wall face 5 z3=0,
Coordinate of wall face 5 x4=20,
Coordinate of wall face 5 y4=0,
Coordinate of wall face 5 z4=0;

The door is 0.91 meters wide and 2.22 meters high. The door position is (0,0.91) meters.
Thus the door,
Coordinate of door face 6 x1=0,
Coordinate of door face 6 y1=0.91,
Coordinate of door face 6 z1=2.22,
Coordinate of door face 6 x2=0,
Coordinate of door face 6 y2=0.91,
Coordinate of door face 6 z2=0,
Coordinate of door face 6 x3=0,
Coordinate of door face 6 y3=0,
Coordinate of door face 6 z3=0,
Coordinate of door face 6 x4=0,
Coordinate of door face 6 y4=0,
Coordinate of door face 6 z4=2.22;

The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (3.63,6.37) meters.
Thus the window,
Coordinate of window face 6 x1=20,
Coordinate of window face 6 y1=3.63,
Coordinate of window face 6 z1=2.41,
Coordinate of window face 6 x2=20,
Coordinate of window face 6 y2=3.63,
Coordinate of window face 6 z2=0.91,
Coordinate of window face 6 x3=20,
Coordinate of window face 6 y3=6.37,
Coordinate of window face 6 z3=0.91,
Coordinate of window face 6 x4=20,
Coordinate of window face 6 y4=6.37,
Coordinate of window face 6 z4=2.41;

The window's U-Factor is 1.5, and its solar heat gain coefficient is 0.3.
The building accommodates 5 people with an activity level of 120, and the occupancy rates are as follows:
60% from Monday to Thursday during 8:00 to 17:00, 80% on Friday from 8:00 to 16:00, 20% on Saturday and Sunday.
The definition for lighting is 8.0 W/m2, and for electrical equipment, it is 10.0 W/m2. The infiltration rate is 2 ACH.
The heating setpoint is 24.0 degrees Celsius, and the cooling setpoint is 26.0 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exter Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
Exterior Finish OSB: 0.015875 meters thick, with a conductivity of 0.119189927357443 W/m-K.
Rigid Insulation: 0.0254 meters thick, with a conductivity of 0.0288455777728565 W/m-K.
OSB: 0.0127 meters thick, with a conductivity of 0.115382311091426 W/m-K.
Stud Layer: 0.1397 meters thick, with a conductivity of 0.0422155030705755 W/m-K.
EPDM Rubber: 0.00254 meters thick, with a conductivity of 0.200318114843602 W/m-K.
Insulted Truss: 0.949325 meters thick, with a conductivity of 0.0493980519360168 W/m-K.
Drywall: 0.015875 meters thick, with a conductivity of 0.160251607317104 W/m-K.
Roddent Barrier: 0.00127 meters thick, with a conductivity of 0.500759230136789 W/m-K.
Insulated Joist: 0.3673348 meters thick, with a conductivity of 0.0479269274696011 W/m-K.
Plywood: 0.00635 meters thick, with a conductivity of 0.115382311091426 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.

Therefore, the IDF code is:

BuildingSurface:Detailed,
  Face 0,
  Floor,
  Floor,
  Thermal Zone 1,
  ,
  Ground,
  ,
  NoSun,
  NoWind,
  ,
  ,
  0, 0, 0,
  0, 10, 0,
  20, 10, 0,
  20, 0, 0;

BuildingSurface:Detailed,
  Face 1,
  Roof,
  Roof,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  20, 10, 3,
  0, 10, 3,
  0, 0, 3;

BuildingSurface:Detailed,
  Face 2,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 0, 3,
  0, 10, 3,
  0, 10, 0,
  0, 0, 0;

BuildingSurface:Detailed,
  Face 3,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  0, 10, 3,
  20, 10, 3,
  20, 10, 0,
  0, 10, 0;

BuildingSurface:Detailed,
  Face 4,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 10, 3,
  20, 0, 3,
  20, 0, 0,
  20, 10, 0;

BuildingSurface:Detailed,
  Face 5,
  Wall,
  Exterior Wall,
  Thermal Zone 1,
  ,
  Outdoors,
  ,
  SunExposed,
  WindExposed,
  ,
  ,
  20, 0, 3,
  0, 0, 3,
  0, 0, 0,
  20, 0, 0;

FenestrationSurface:Detailed,
  Face 7,
  Door,
  Door,
  Face 2,
  ,
  ,
  ,
  ,
  ,
  0, 0.91, 2.22,
  0, 0.91, 0,
  0, 0, 0,
  0, 0, 2.22;

FenestrationSurface:Detailed,
  Face 8,
  Window,
  Window,
  Face 4,
  ,
  ,
  ,
  ,
  ,
  20, 3.63, 2.41,
  20, 3.63, 0.91,
  20, 6.37, 0.91,
  20, 6.37, 2.41;

Schedule:Day:Interval,Zone1 HtgSetp Default Schedule,Temperature 11,No,24:00,24.0;
Schedule:Day:Interval,Zone1 ClgSetp Default Schedule,Temperature 4,No,24:00,26.0;
Material,Drywall,MediumSmooth,0.015875,0.160251607317104,799.962060895569,1089.991512,0.9,0.7,0.7;
Material,EPDM Rubber,Smooth,0.00254,0.200318114843602,1009.16319255949,2009.664,0.9,0.3,0.7;
Material,Exterior Finish OSB,Smooth,0.015875,0.119189927357443,543.987016179686,1209.9852,0.9,0.5,0.7;
Material,G01a 19mm gypsum board,MediumSmooth,0.019,0.16,800,1090,0.9,0.4,0.4;
Material,Insulated Joist,Smooth,0.3673348,0.0479269274696011,101.492983937411,986.82876,0.9,0.7,0.7;
Material,Insulted Truss,Smooth,0.949325,0.0493980519360168,52.7968552805726,795.492,0.9,0.7,0.7;
Material,OSB,Smooth,0.0127,0.119189927357443,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Plywood,Smooth,0.00635,0.115382311091426,543.987016179686,1209.9852,0.9,0.7,0.7;
Material,Rigid Insulation,Smooth,0.0254,0.0288455777728565,36.4996706439056,1498.8744,0.9,0.7,0.7;
Material,Roddent Barrier,Smooth,0.00127,0.500759230136789,949.894878075836,1820.00196,0.9,0.7,0.7;
Material,Stud Layer,Smooth,0.1397,0.0422155030705755,120.795232303033,1077.26364,0.9,0.7,0.7;
WindowMaterial:SimpleGlazingSystem,Window Material Simple Glazing System 1,1.5,0.3,0.42;
Schedule:Day:Interval,Activity Default Schedule,ActivityLevel,No,24:00,120;
Schedule:Day:Interval,Schedule Friday,Fractional,No,08:00,0,16:00,0.8,24:00,0;
Schedule:Day:Interval,Schedule Monday-Thursday,Fractional,No,08:00,0,17:00,0.6,24:00,0;
Schedule:Day:Interval,Schedule Saturday-Sunday,Fractional,No,24:00,0.2;
Lights,Lights 1,Space Type 1,Schedule Ruleset,Watts/Area,,8.0,,,,,1,General;
People,People 1,Space Type 1,Schedule Ruleset,People,5,,,0.3,,People Activity;
ElectricEquipment,Electric Equipment 1,Space Type 1,Schedule Ruleset,Watts/Area,,10.0,,,,,General;
ZoneInfiltration:DesignFlowRate,Infiltration Setting,Space Type 1,Infil Quarter On,AirChanges/Hour,,,,2,0.03,0.003,0,;
"""},
    {"role": "user", "content": """
Based on the above generated structure, simulate a building that is 8.23 meters long, 4.27 meters wide, and 3.35 meters high.
There is a door on the south-west side. The door is 0.91 meters wide and 2.22 meters high, with its bottom-left corner positioned at (0,0.91) meters from the south-west corner of the building.
There is a window on the east side. The window is 2.74 meters wide and 1.5 meters high. The window sill is 0.91 meters high, and the jamb position is (0.765,3.505) meters.

The window's U-Factor is 1.266253, and its solar heat gain coefficient is 0.43.
The building accommodates 3 people with an activity level of 132, and the occupancy rates are as follows:
66.6% from Monday to Thursday during 8:00 to 17:00, 100% on Friday from 8:00 to 16:00, 0% on Saturday and Sunday.
The definition for lighting is 6 W/m2, and for electrical equipment, it is 4 W/m2. The infiltration rate is 1.4 ACH.
The heating setpoint is 22.2.0 degrees Celsius, and the cooling setpoint is 24.4 degrees Celsius.

For constructions
The layers of the exterior wall form outside to inside are: Exterior Finish OSB, Rigid Insulation, OSB, Stud Layer.
The layers of the roof form outside to inside are: EPDM Rubber, OSB, Insulted Truss, Drywall.
The layers of the floor from outside to inside are: Roddent Barrier, Insulated Joist, OSB, Plywood.
The layers of the interior wall form outside to inside are: G01a 19mm gypsum board, G01a 19mm gypsum board.

For materials
exterior Finish OSB: 0.02 meters thick, with a conductivity of 0.22 W/m-K.
Rigid Insulation: 0.03 meters thick, with a conductivity of 0.05 W/m-K.
OSB: 0.01 meters thick, with a conductivity of 0.18 W/m-K.
Stud Layer: 0.15 meters thick, with a conductivity of 0.048 W/m-K.
EPDM Rubber: 0.0028 meters thick, with a conductivity of 0.25 W/m-K.
Insulted Truss: 0.90 meters thick, with a conductivity of 0.05 W/m-K.
Drywall: 0.02 meters thick, with a conductivity of 0.18 W/m-K.
Roddent Barrier: 0.002 meters thick, with a conductivity of 0.5 W/m-K.
Insulated Joist: 0.4 meters thick, with a conductivity of 0.05 W/m-K.
Plywood: 0.008 meters thick, with a conductivity of 0.22 W/m-K.
G01a 19mm gypsum board: 0.019 meters thick, with a conductivity of 0.16 W/m-K.

Let's think step by step.
"""}
] 

### division
real_reason_shot_prompt = tokenizer.apply_chat_template(real_reason_shot, tokenize=False, add_generation_prompt=True)
real_reason_shot_prompt_inputs = tokenizer(real_reason_shot_prompt, return_tensors="pt").to(model.device)
real_reason_shot_prompt_outputs = model.generate(**real_reason_shot_prompt_inputs, use_cache=True, max_length=10000)
real_reason_shot_prompt_output_text = tokenizer.decode(real_reason_shot_prompt_outputs[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("real_division")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(real_reason_shot_prompt_output_text)
