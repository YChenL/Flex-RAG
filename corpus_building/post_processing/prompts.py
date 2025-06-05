CAPTION_PROMPT = """
You will be given a scientific image. 

1. **First**, decide which one of the following five categories it belongs to:
   - statistical_plot  
   - visualization_figure  
   - pipeline_diagram  
   - table
   - other_image  

2. **Then**, generate a detailed, precise description according to the category you chose, following these templates:

- **If statistical_plot**  
  1. Overview: Describe the purpose of the plot, including what is being measured and compared.  
  2. Axes Explanation: Describe the meaning of each axis, including respective logarithmic scales.  
  3. Legend Interpretation: Explain what each curve/line represents and how each one performs relative to each other.  
  4. Trends Description: Highlight significant distributions or trends, focusing on rates of change, inflection points, or other significant features.

- **If visualization_figure**  
  1. Overview: Introduce the purpose of the visualization, explaining what kind of data or experimental results are being presented.  
  2. Key Features: Describe the key visual components (axes, colors, markers, or spatial distributions) and explain their significance.  
  3. Methodological Insights: If applicable, explain how different methods, models, or experimental conditions are being compared.  
  4. Implications: Discuss the conclusions that can be drawn directly from the visualization; do not fabricate conclusions without supporting evidence.

- **If pipeline_diagram**  
  1. Overall Description: Summarize the goal of the pipeline, its major modules, and how they work together.  
  2. Step-by-Step Explanation: Describe the sequence of operations.  
  3. Key Modules and Functions: Highlight important elements such as backbone, data inputs/outputs, and main algorithms.  
  4. Diagram Details: Explain how colors, arrows, and blocks represent operations and data flow.

- **If table**  
  1. Table Overview: Describe the purpose of the table, including key information and research context.  
  2. Content Description: Explain what the table represents (experimental results, statistical summaries, parameter comparisons, etc.).  
  3. Structure Explanation: Describe what each row and column represent.  
  4. Key Findings: Highlight notable trends, relationships, or interpretations.
  
- **If other_image**  
  1. Overview: Describe the purpose of the image (the concepts, objects, or phenomena it represents).  
  2. Scientific Context: Explain which aspect of the study it illustrates.  
  3. Observations: Highlight notable details, relationships, or interpretations that can be drawn directly; do not fabricate conclusions.

**Output** only the final caption after you’ve chosen the right category.
"""