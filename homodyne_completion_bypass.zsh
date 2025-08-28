#!/usr/bin/env zsh
# Homodyne Shell Completion - Bypass compdef issues
# This uses zsh's programmable completion directly

# Create the completion function for homodyne
_homodyne_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")
    
    # Get the previous word - fix the parsing logic
    local prev=""
    if (( ${#words} > 1 )); then
        # If the line ends with a space, we're completing after the last word
        if [[ "$line" =~ ' $' ]]; then
            prev="${words[-1]}"
        else
            # Otherwise we're in the middle of a word, so previous is words[-2]
            prev="${words[-2]}"
        fi
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
        --install-completion|--uninstall-completion)
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
                    --uninstall-completion
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

# Create the completion function for homodyne-config
_homodyne_config_complete() {
    local cur="${BUFFER##* }"
    local line="$BUFFER"
    local words=("${(@)${(z)line}}")
    
    # Get the previous word - fix the parsing logic
    local prev=""
    if (( ${#words} > 1 )); then
        # If the line ends with a space, we're completing after the last word
        if [[ "$line" =~ ' $' ]]; then
            prev="${words[-1]}"
        else
            # Otherwise we're in the middle of a word, so previous is words[-2]
            prev="${words[-2]}"
        fi
    fi
    
    # Generate completions based on context
    local -a completions
    
    case "$prev" in
        --mode|-m)
            completions=(static_isotropic static_anisotropic laminar_flow)
            ;;
        --output|-o)
            completions=(*.json(N) config/*.json(N) configs/*.json(N))
            ;;
        --sample|-s|--experiment|-e|--author|-a)
            # These don't have specific completions, just return empty
            completions=()
            ;;
        *)
            if [[ "$cur" == --* ]]; then
                completions=(
                    --mode
                    --output
                    --sample
                    --experiment
                    --author
                    --help
                )
            elif [[ "$cur" == -* ]]; then
                completions=(
                    -m
                    -o
                    -s
                    -e
                    -a
                    -h
                )
            else
                # No positional arguments for homodyne-config
                completions=()
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
zle -C _homodyne_config_complete_widget complete-word _homodyne_config_complete

# Bind the widget to a key sequence
# This creates a manual completion that works when compdef fails
bindkey '^Xh' _homodyne_complete_widget
bindkey '^Xc' _homodyne_config_complete_widget

# Create convenient aliases as completion alternatives
alias hc='homodyne --method classical'
alias hm='homodyne --method mcmc'
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# Config file shortcuts
alias hconfig='homodyne --config'
alias hplot='homodyne --plot-experimental-data'

# homodyne-config shortcuts
alias hc-iso='homodyne-config --mode static_isotropic'
alias hc-aniso='homodyne-config --mode static_anisotropic'  
alias hc-flow='homodyne-config --mode laminar_flow'
alias hc-config='homodyne-config'

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
    echo "homodyne-config shortcuts:"
    echo "  hc-iso    = homodyne-config --mode static_isotropic"
    echo "  hc-aniso  = homodyne-config --mode static_anisotropic"
    echo "  hc-flow   = homodyne-config --mode laminar_flow"
    echo "  hc-config = homodyne-config"
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

# For homodyne-config, compdef has issues with the dash, so use compctl as fallback
if ! compdef _homodyne_config_complete homodyne-config 2>/dev/null; then
    # Use compctl as fallback for commands with dashes
    if compctl -K _homodyne_config_complete homodyne-config 2>/dev/null; then
        # Successfully registered with compctl
        true
    else
        # If both compdef and compctl fail, provide manual completion
        echo "Note: Automatic completion for homodyne-config may not work."
        echo "Use Ctrl-X followed by 'c' for manual completion, or use these shortcuts:"
        echo "  homodyne-config --mode static_isotropic"
        echo "  homodyne-config --mode static_anisotropic" 
        echo "  homodyne-config --mode laminar_flow"
    fi
fi