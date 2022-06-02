<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>


# A Gurobi Wrapper that Supports Numpy Operations

Module `tensorgrb.py` fully supports constructing, combining, manipulating Gurobi variables and constraints using numpy arrays. This interface is convenient for setting up matrix-based formulations. However, for super-large-size models, the current version is slow. 


The other module `tensorgp.py` uses Gurobi's native matrix api. The construction time is much faster than `tensorgrb,py`, but the matrix operations are very limited.

## Module `tensorgrb.py` 
The follow is the formulation to be constructed.

\begin{align}
\min_{i \in I} 
\end{align}
