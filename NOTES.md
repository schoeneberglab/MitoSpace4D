# 4D Mitospace Notes
## Eric Arkfeld

### To-Do's
- [] Set up .env and replace hard-coded paths?
- [] Set up experiment-specific color/drug config files?


### 20250827
- Plotting 3D frame embeddings generated using 4D mitospace (processed as independent single-frame movies)
- Set up visualization routines for retrieving picked frames and coloring by drug as well as time by region as well as absolute frame within the imaging session 

Observations
- Consecutive frames for cells appear as paths through the down projected space
- Central cluster appears structured with paths tracing around in the structure with some extending out radially.
- Dead cells appear to diverge radially from the center
  
Experiments
- Identify dead cells, filter those embeddings and reproject?



### 20250818

Evaluating trends in kinetic mitospace w/ temporal colormap:

- 13 mitomycinc - unclear
- 5 h2o2 - clear trend
- 3 mfi8 - unclear
- 15 latrinculinb - clear trend
- 16 mdivi1 - relatively clear trend
- 8 lonidamine - unclear
- 14 cytochalasind - somewhat of a trend
- 10 dnp - relatively unclear
- 12 cccp - relatively unclear
- 11 valinomycin - relatively unclear

$\rightarrow$ Try plotting vectors to visualize instance-level movement through the space? \
$\rightarrow$ Try generating 60-frame embeddings? Or plot only the first timepoint?
