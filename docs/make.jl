using Documenter
using MPSTime


makedocs(
    sitename = "MPSTime",
    format = Documenter.HTML(),
    modules = [MPSTime],
    pages = [
    "Home" => "index.md",
    "Tutorial: Classification" => "tutorial.md",
    "Imputation" => "imputation.md",
    "Encodings" => "encodings.md",
    "Docstrings" => "docstrings.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

# testing locally
# julia --color=yes --project make.jl

# julia -e 'using LiveServer; serve(dir="docs/build")'
