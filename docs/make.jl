using Documenter
using MPSTime


makedocs(
    sitename = "MPSTime",
    format = Documenter.HTML(sidebar_sitename=false),
    modules = [MPSTime],
    pages = [
    "Home" => "index.md",
    "Tutorial: Classification" => "tutorial.md",
    "Imputation" => "imputation.md",
    "Encodings" => "encodings.md",
    "Docstrings" => "docstrings.md"
    ]
)

deploydocs(
    repo = "github.com/jmoo2880/MPSTime.jl.git",
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
