using Documenter
using MPSTime


makedocs(
    sitename = "MPSTime",
    format = Documenter.HTML(),
    modules = [MPSTime]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

# julia -e 'using LiveServer; serve(dir="docs/build")'
# julia --color=yes --project make.jl
