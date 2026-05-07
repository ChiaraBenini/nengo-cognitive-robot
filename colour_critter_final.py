import grid
import nengo
import nengo.spa as spa
import numpy as np 


#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#########
#  M   R#
#R#R#B#R#
# # # # #
#G Y   R#
#########
"""



#### Preliminaries - this sets up the agent and the environment ################ 
class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=int(4))

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

noise_val = 0.01 # how much noise there will be in the colour info

#################My Code ############################################
#here I'm pre-computing the opponents (same as done for the sensors) outside of the model

def rgb_to_opponent(rgb):
    M = np.array([
        [ 1.0, -1.0,  0.0],   # R - G
        [-0.5, -0.5,  1.0],   # B - (R+G)/2
        [ 0.5,  0.5,  0.0],   # luminance
    ]).T
    return rgb @ M


color_prototypes = {}

for cid, rgb in col_values.items():
    if cid == 0:
        continue  # ignoring white
    opp = rgb_to_opponent(np.array(rgb))
    proto = np.concatenate([opp, [0.0]])  # padding to 4d to match what i already jave
    proto /= np.linalg.norm(proto)         # normalize for similarity 
    color_prototypes[cid] = proto



#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()
with model:
    
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)

    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        
    movement = nengo.Node(move, size_in=2)
    
    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #--------------------------------------------------------------------------#
    def cell2rgb(t):
        
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        # minimal white-handling fix: snap nearly-white to exact white
        if np.linalg.norm(c - np.array([0.9, 0.9, 0.9])) < 0.05:
            c = np.array([0.9, 0.9, 0.9])
        
        return c
        
    current_color = nengo.Node(cell2rgb)
  
     
    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-white       #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    def look_ahead(t):
        
        done = False
        
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
            
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            
            if cell.cellcolor > 0:
                done = True
        
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        # minimal white-handling fix
        if np.linalg.norm(c - np.array([0.9, 0.9, 0.9])) < 0.05:
            c = np.array([0.9, 0.9, 0.9])
        
        return c
        
    ahead_color = nengo.Node(look_ahead)    
    
    ### Agent functionality - your code adds to this section ###################
    
    #All input nodes should feed into one ensemble. Here is how to do this for
    #the radar, see if you can do it for the others
    walldist = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)

    #--------------------------------------------------------------------------#
    # For now, all our agent does is wall avoidance. It uses values of the radar #
    # to: a) turn away from walls on the sides and b) slow down in function of  #
    # the distance to the wall ahead, reversing if it is really close           #
    #--------------------------------------------------------------------------#
    # movement_func now expects 4D input: 3 radar + 1 curiosity
    def movement_func(x):
        left, center, right, curiosity = x


        # it keeps getting stuck so IM doing an emergency break condition
        if center < 0.15:  # Very close to wall)
            # Sharp turn away from  side wt more space
            if left > right:
                turn = -0.8  # Sharp left turn
            else:
                turn = 0.8   # Sharp right turn
            spd = -0.2  # Back up a bit while turning !! this otherwise with different or small maps it get stuck so much
            return spd, turn
        

        # Wall avoidance (kept as original, jsut renamed variables)
        turn = right - left
        
        
        # if we dont scale curiosity the agent ends up being super slow all the time, and gets stuck (undecided)
        if center < 0.3:  # Too close to wall
            curiosity_scaled = curiosity * 0.1  # super low near walls
        elif center < 0.6:  # mediuum distance
            curiosity_scaled = curiosity * 0.7
        else:  
            curiosity_scaled = curiosity * 1.5  # Strong when safe/clear parth
        
        # Apply curiosity as additive forsce(negative means steer away)
        turn += curiosity_scaled
        
        # Small random component to break loops (i tested some larger maps with less obstacles and it would keep going in circles at times)
        turn += np.random.normal(0, 0.03)
        
        # we dont wan extreme turns
        turn = np.clip(turn, -0.5, 0.5)
        
        # Speed: slower near walls, this is also in line with logic of the orignal movemnt
        if center < 0.3:
            spd = 0.3
        elif center < 0.6:
            spd = 0.6
        else:
            spd = 0.9
        
        return spd, turn

    #--------------------------------------------------------------------------#
    # Layer 1: RGB encoding into neural representation                         #
    # this is the part where we transform the sensor signals into ensebles, so #
    # its all converted in a neural signal.                                    #
    #--------------------------------------------------------------------------#
    current_rgb_ens = nengo.Ensemble(n_neurons=300, dimensions=3, radius=1.5, label="current_rgb")
    nengo.Connection(current_color, current_rgb_ens)
    
    ahead_rgb_ens = nengo.Ensemble(n_neurons=300, dimensions=3, radius=1.5, label="ahead_rgb")
    nengo.Connection(ahead_color, ahead_rgb_ens)
    
    #--------------------------------------------------------------------------#
    # Layer 2: opponent color transformation                                   #
    # this is the layer that does the axon categorizion, corresponst to Nucleo #
    #Genicolato Laterale. It makes the color categorization oriente in opposing#
    #axes based on components contrasts. Taking the math from my neuroscience. #
    #book and making it matrix-like
    #--------------------------------------------------------------------------#
    current_opponent = nengo.Ensemble(n_neurons=300, dimensions=3, radius=2.5, label="Current Opponent")
    ahead_opponent = nengo.Ensemble(n_neurons=300, dimensions=3, radius=2.5, label="Ahead Opponent")
    
    opponent_transform = np.array([
        [ 1.0, -1.0,  0.0],    # R - G
        [-0.5, -0.5,  1.0],    # B - (R+G)/2
        [ 0.5,  0.5,  0.0],    # (R+G)/2 (luminance)
    ]) 
    
    nengo.Connection(current_rgb_ens, current_opponent, transform=opponent_transform)
    nengo.Connection(ahead_rgb_ens, ahead_opponent, transform=opponent_transform)

    #--------------------------------------------------------------------------#
    # Layer 3: cortical color space                                            #
    #converts axon to space/manifold color categorization. theoretically in    #
    # downstream the colors would emerge organically and then we would 'name'  #
    # in a more top down way, thorugh language processing. I added a dimension #
    # to do more with it later on, but i didnt have the tiime to implement that#
    # so its set to 0
    #--------------------------------------------------------------------------#
    tau_stabilize = 0.01 #using to stabiiliize later,as a synapse arg. means that it stabilizes over 10ms (smoothing)
    current_cortex = nengo.Ensemble(n_neurons=600, dimensions=4, radius=2.0, label="current_cortex")
    ahead_cortex = nengo.Ensemble(n_neurons=600, dimensions=4, radius=2.0, label="ahead_cortex")
    
    nengo.Connection(current_opponent, current_cortex, synapse=tau_stabilize, transform=[[1,0,0],[0,1,0],[0,0,1],[0,0,0]]) #going from 3d to 4d, keeping it 0 for now
    nengo.Connection(ahead_opponent, ahead_cortex, synapse=tau_stabilize, transform=[[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    
    def normalize(t, x): #normalizing by magnitude so its easier to compare later (gave me issues without normalization)
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-6 else np.zeros_like(x)
    
    current_cortex_norm = nengo.Node(normalize, size_in=4, size_out=4, label="current_cortex_norm")
    ahead_cortex_norm = nengo.Node(normalize, size_in=4, size_out=4, label="ahead_cortex_norm")
    
    nengo.Connection(current_cortex, current_cortex_norm, synapse=0.01)
    nengo.Connection(ahead_cortex, ahead_cortex_norm, synapse=0.01)

    #--------------------------------------------------------------------------#
    # Similarity-based color comparison: this is the part where we flag down   #
    # which color is which. labelling
    #--------------------------------------------------------------------------#
    color_names = {
        1: "GREEN",
        2: "RED",
        3: "BLUE",
        4: "MAGENTA",
        5: "YELLOW",
    }
    
   
    color_similarity = {}
    
    for cid, name in color_names.items():
        ens = nengo.Ensemble(
            n_neurons=180 if name == "RED" else 150,  # giving more neurons to red since its more relevant
            dimensions=1,
            radius=1.0, #expets inputi betwwen -1/1, to be in renage with cosiine similiarity
            label=f"{name}_similarity_current"
        )
    
        proto = color_prototypes[cid]  # explicitly defined
    
        # cosine similarity (because cortex is normalized)
        nengo.Connection(
            current_cortex_norm, #4d
            ens, #1d
            transform=proto.reshape(1, -1), #dot product matrix
            synapse=0.01 #stabilize
        )
    
        color_similarity[cid] = ens

    # vector to put current color similarities together
    current_color_sim_vector = nengo.Node(size_in=5, label="current_color_similarity_vector")
    
    nengo.Connection(color_similarity[1], current_color_sim_vector[0])  # GREEN
    nengo.Connection(color_similarity[2], current_color_sim_vector[1])  # RED
    nengo.Connection(color_similarity[3], current_color_sim_vector[2])  # BLUE
    nengo.Connection(color_similarity[4], current_color_sim_vector[3])  # MAGENTA
    nengo.Connection(color_similarity[5], current_color_sim_vector[4])  # YELLOW
    
    # same identical thing for the ahead stream
    ahead_color_similarity = {}
    
    for cid, name in color_names.items():
        ens = nengo.Ensemble(
            n_neurons=180 if name == "RED" else 150,
            dimensions=1,
            radius=1.0,
            label=f"{name}_similarity_ahead"
        )
        proto = color_prototypes[cid]
        nengo.Connection(
            ahead_cortex_norm,
            ens,
            transform=proto.reshape(1, -1),
            synapse=0.01
        )
        ahead_color_similarity[cid] = ens

    # vector to put ahead color similarities together
    ahead_color_sim_vector = nengo.Node(size_in=5, label="ahead_color_similarity_vector")
    
    nengo.Connection(ahead_color_similarity[1], ahead_color_sim_vector[0])  # GREEN
    nengo.Connection(ahead_color_similarity[2], ahead_color_sim_vector[1])  # RED
    nengo.Connection(ahead_color_similarity[3], ahead_color_sim_vector[2])  # BLUE
    nengo.Connection(ahead_color_similarity[4], ahead_color_sim_vector[3])  # MAGENTA
    nengo.Connection(ahead_color_similarity[5], ahead_color_sim_vector[4])  # YELLOW
    

    ##wta##
    #now to pick the color we do a winner takes all type of choice
    def classify_color(t, x):
        # x = similarity vector for 5 colors
        y = np.zeros(5)
        
        # check if all similarities are below threshold 
        if np.max(x) < 0.4:  
            return y  #its for ambiguity and noise
        
        # pick the highest similarity
        idx = np.argmax(x)
        y[idx] = 1.0
        return y #basically we get back a 5d vecotr (one d for each color) and everything is 0 except the color that matches (1)

    
    current_color_id = nengo.Node(
        classify_color,
        size_in=5,
        size_out=5,
        label="current_color_id"
    )
    nengo.Connection(current_color_sim_vector, current_color_id, synapse=0.01)

    ahead_color_id = nengo.Node(
        classify_color,
        size_in=5,
        size_out=5,
        label="ahead_color_id"
    )
    nengo.Connection(ahead_color_sim_vector, ahead_color_id, synapse=0.01)

    #two classifier nodes for each stream

    #--------------------------------------------------------------------------#
    # Previous color memory     : doing reccurent connection with 90% memoery retention per timestep
    # its aslo slow (syn = 0.1) so the memroy should persist longer, be more stable (?not sure how to test this)                                               #
    #--------------------------------------------------------------------------#
    prev_color_id = nengo.Ensemble(n_neurons=300, dimensions=5, radius=1.5, label="prev_color_id")
    nengo.Connection(prev_color_id, prev_color_id, transform=0.9, synapse=0.1)
    nengo.Connection(current_color_id, prev_color_id, synapse=0.05)

    #--------------------------------------------------------------------------#
    # Transition space: prev current  25D                                     #
    #--------------------------------------------------------------------------#
    def transition_space(t,x):
        prev = x[:5] 
        curr = x[5:]
        return np.outer(prev,curr).ravel() #ravel is to flatten, outer is to do the outer product(5x5 matrix)

    #inidex 5 is where red is, so we only care of the transition happening on certain spot
    transition_node = nengo.Node(transition_space, size_in=10, size_out=25, label="transition_space")
    nengo.Connection(prev_color_id, transition_node[:5], synapse=None)
    nengo.Connection(current_color_id, transition_node[5:], synapse=None)

    #--------------------------------------------------------------------------#
    # Color change detector                                                     #
    #--------------------------------------------------------------------------#
    def detect_color_change(t,x):
        prev = x[:5]
        curr = x[5:]
        if np.sum(curr) > 0 and np.argmax(prev) != np.argmax(curr): #this shoudl return trie if diff colors and false if same
            return 1.0
        return 0.0

    color_change = nengo.Node(detect_color_change, size_in=10, size_out=1, label="color_change")
    nengo.Connection(prev_color_id, color_change[:5], synapse=None)
    nengo.Connection(current_color_id, color_change[5:], synapse=None)

    #--------------------------------------------------------------------------#
    # RED/X transition counters                                                #
    #--------------------------------------------------------------------------#
    RED = 1
    transition_indices = {"GREEN":RED*5+0,"BLUE":RED*5+2,"MAGENTA":RED*5+3,"YELLOW":RED*5+4} #here is where the index 5 thing comes into play
    counters = {}
    counter_outputs = {}  #  storing

    #we're creating the same enslbles and connections for all the possible transitions (4)
    for name, idx in transition_indices.items():
        gate = nengo.Ensemble(n_neurons=60, dimensions=1, radius=1.0, label=f"{name}_gate")#this shoudl work like and and gate, and activate when a specific transition occurs and color change occurs (0 or1)
        counter = nengo.Ensemble(n_neurons=200, dimensions=1, radius=10, label=f"Count RED→{name}") #integrator, accumaletes countrsover time (up to 10 transitiions). large because it needs preciousio, plus reccurent connection for memory
        
        # Store counter output in a separate node, to access them later
        counter_output = nengo.Node(size_in=1, label=f"{name}_count_output")
        nengo.Connection(counter, counter_output, synapse=None)
        counter_outputs[name] = counter_output #storing info as dict
        
        #connecting eveuthing with also memory when necessary
        nengo.Connection(transition_node[idx], gate, synapse=0.01)
        nengo.Connection(color_change, gate, synapse=0.01)
        nengo.Connection(gate, counter, synapse=0.05)
        nengo.Connection(counter, counter, transform=0.99, synapse=0.1)
        counters[name] = counter

    #--------------------------------------------------------------------------#
    # Curiosity Node (i tested this and changed this sooo many times)                                         #
    #--------------------------------------------------------------------------#
    def curiosity_bias(t, x):
        current_id = x[:5]    # One-hot current color
        ahead_id = x[5:10]    # One-hot ahead color
        counts = x[10:]       # RED X counts 
        
        RED_IDX = 1  # to find the red in the vector
        
        # Only curious when CURRENt!!! color is RED
        if current_id[RED_IDX] > 0.5:
            # Find  out which non-red color is ahead
            ahead_idx = np.argmax(ahead_id)
            
            # Map ahead color to counter index
            # 0:GREEN, 2:BLUE, 3:MAGENTA, 4:YELLOW 
            mapping = {0: 0, 2: 1, 3: 2, 4: 3}
            counter_idx = mapping.get(ahead_idx)
            
            if counter_idx is not None and len(counts) > counter_idx:
                total = np.sum(counts)
                if total > 0:
                    # Familiarity = how often we've made this transition
                    familiarity = counts[counter_idx] / total #(normalized 0 to 1)
                    
                    # tuned this part several time, can change it but risk slowwer movements from agent,
                    #or that they get stuck. this should be fine for now
                    bias = -0.6 * familiarity #negative sign because we're steering away from familiarity, thus implementing curiosity
                    return bias
        
        return 0.0 #if not red to x transition

    #  node to collect all counter values, then one each. keep them separete for debugging and monitoring
    counter_values_node = nengo.Node(size_in=4, label="counter_values")
    
    # Connect counter outputs to the counter values node
    nengo.Connection(counter_outputs["GREEN"], counter_values_node[0], synapse=0.01)
    nengo.Connection(counter_outputs["BLUE"], counter_values_node[1], synapse=0.01)
    nengo.Connection(counter_outputs["MAGENTA"], counter_values_node[2], synapse=0.01)
    nengo.Connection(counter_outputs["YELLOW"], counter_values_node[3], synapse=0.01)

    # Create curiosity node with all things
    curiosity_node = nengo.Node(
        curiosity_bias, 
        size_in=14,  # 5 (current) + 5 (ahead) + 4 (counts) = 14
        size_out=1, 
        label="curiosity_bias"
    )
    
    # Connect all inputs to curiosity node
    nengo.Connection(current_color_id, curiosity_node[:5], synapse=None)
    nengo.Connection(ahead_color_id, curiosity_node[5:10], synapse=None)
    nengo.Connection(counter_values_node, curiosity_node[10:], synapse=0.01)

    #--------------------------------------------------------------------------#
    # Combine movement function and distance sensory wit the curiosity thing                                               #
    #--------------------------------------------------------------------------#
    combined_input = nengo.Ensemble(n_neurons=100, dimensions=4)
    nengo.Connection(walldist, combined_input[:3])
    nengo.Connection(curiosity_node, combined_input[3])
    nengo.Connection(combined_input, movement, function=movement_func)