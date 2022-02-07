using TeukolskyQNMFunctions
using Documenter

DocMeta.setdocmeta!(TeukolskyQNMFunctions, :DocTestSetup, :(using TeukolskyQNMFunctions); recursive=true)

makedocs(;
    modules=[TeukolskyQNMFunctions],
    authors="Justin L. Ripley",
    repo="https://github.com/JLRipley314/TeukolskyQNMFunctions.jl.git",
    sitename="TeukolskyQNMFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JLRipley314.github.io/TeukolskyQNMFunctions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JLRipley314/TeukolskyQNMFunctions.jl",
    devbranch="main",
)
