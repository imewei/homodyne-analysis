#!/usr/bin/env zsh
# Homodyne Shell Completion - Bypass compdef issues
# This uses zsh's programmable completion directly

# Create the completion function
_homodyne_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")
    
    # Get the previous word
    local prev=""
    if (( ${#words} > 1 )); then
        prev="${words[-2]}"
    fi
    
    # Generate completions based on context
    local -a completions
    
    case "$prev" in
        --method)
            completions=(classical mcmc robust all)
            ;;
        --config)
            completions=(*.json(N) config/*.json(N) configs/*.json(N))
            ;;
        --output-dir)
            completions=(*/(N))
            ;;
        --install-completion)
            completions=(bash zsh fish powershell)
            ;;
        *)
            if [[ "$cur" == --* ]]; then
                completions=(
                    --method
                    --config
                    --output-dir
                    --verbose
                    --quiet
                    --static-isotropic
                    --static-anisotropic
                    --laminar-flow
                    --plot-experimental-data
                    --plot-simulated-data
                    --contrast
                    --offset
                    --phi-angles
                    --install-completion
                )
            else
                completions=(*)
            fi
            ;;
    esac
    
    # Filter completions based on current input
    if [[ -n "$cur" ]]; then
        completions=(${(M)completions:#${cur}*})
    fi
    
    # Set up completion
    if (( ${#completions} > 0 )); then
        compadd -a completions
    fi
}

# Create a widget for the completion
zle -C _homodyne_complete_widget complete-word _homodyne_complete

# Bind the widget to a key sequence
# This creates a manual completion that works when compdef fails
bindkey '^Xh' _homodyne_complete_widget

# Create convenient aliases as completion alternatives
alias hc='homodyne --method classical'
alias hm='homodyne --method mcmc'
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# Config file shortcuts
alias hconfig='homodyne --config'
alias hplot='homodyne --plot-experimental-data'

# Also create a simple completion helper function
homodyne_help() {
    echo "Homodyne command completions:"
    echo ""
    echo "Method shortcuts:"
    echo "  hc  = homodyne --method classical"
    echo "  hm  = homodyne --method mcmc" 
    echo "  hr  = homodyne --method robust"
    echo "  ha  = homodyne --method all"
    echo ""
    echo "Other shortcuts:"
    echo "  hconfig = homodyne --config"
    echo "  hplot   = homodyne --plot-experimental-data"
    echo ""
    echo "Available methods: classical mcmc robust all"
    echo "Config files in current dir:"
    local configs=(*.json(N))
    if (( ${#configs} > 0 )); then
        printf "  %s\n" "${configs[@]}"
    else
        echo "  (no .json files found)"
    fi
    echo ""
    echo "Common flags: --verbose --quiet --static-isotropic --static-anisotropic --laminar-flow"
}

# Try compdef registration, but don't fail if it doesn't work
# (Silent registration - no startup messages)
compdef _homodyne_complete homodyne 2>/dev/null