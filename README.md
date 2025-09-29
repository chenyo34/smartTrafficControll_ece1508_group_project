# smartTrafficControll_
Group Project for ECE1508H 2025 Fall


### States/Observation Space (not confirmed by group yet)
- Traffic lights
  - Color 
- Lanes
  - Fixed 
- Vehicles
  - Velocity
  - Position 


### Terminate states (not confirmed by group yet)
- Lanes fully occupied by vechciles
- Time Limit

### Action Space (not confirmed by group yet)
- Traffic light switching


### Reward mechanism/functions (not confirmed by group yet)
The Reward is a complex function that influences the efficiency of the traffic light
- The available capacity of the Lanes
  - A ratio => ( # of vehicles in lane X / # total vehicles ) / ( # max capacity of lane X/ (# global capacity for all lanes ) 
- Vehicles
  - Stopped Vehicles
  - Slowing-down Vehicles
  - Passing Vehicles
  - Average Speed





