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

# Create the completion function for homodyne-gpu
_homodyne_gpu_complete() {
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

    # homodyne-gpu only supports mcmc and all methods (classical/robust show error)
    local -a completions

    case "$prev" in
        --method)
            completions=(mcmc all)
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


# Create convenient aliases as completion alternatives
alias hc='homodyne --method classical'
alias hm='homodyne --method mcmc'
alias hr='homodyne --method robust'
alias ha='homodyne --method all'

# homodyne-gpu shortcuts (only supports mcmc and all)
alias hgm='homodyne-gpu --method mcmc'
alias hga='homodyne-gpu --method all'

# Deprecated aliases that show helpful errors
alias hgc='echo "❌ homodyne-gpu --method classical not supported. Use: homodyne --method classical" && false'
alias hgr='echo "❌ homodyne-gpu --method robust not supported. Use: homodyne --method robust" && false'

# Config file shortcuts
alias hconfig='homodyne --config'
alias hplot='homodyne --plot-experimental-data'
alias hgconfig='homodyne-gpu --config'

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
    echo "homodyne-gpu shortcuts (GPU acceleration, Linux only):"
    echo "  hgm = homodyne-gpu --method mcmc"
    echo "  hga = homodyne-gpu --method all"
    echo ""
    echo "Note: homodyne-gpu only supports mcmc/all methods"
    echo "      For classical/robust, use regular homodyne command"
    echo ""
    echo "Other shortcuts:"
    echo "  hconfig  = homodyne --config"
    echo "  hgconfig = homodyne-gpu --config"
    echo "  hplot    = homodyne --plot-experimental-data"
    echo ""
    echo "homodyne-config shortcuts:"
    echo "  hc-iso    = homodyne-config --mode static_isotropic"
    echo "  hc-aniso  = homodyne-config --mode static_anisotropic"
    echo "  hc-flow   = homodyne-config --mode laminar_flow"
    echo "  hc-config = homodyne-config"
    echo ""
    echo "Available methods:"
    echo "  homodyne: classical mcmc robust all (all methods)"
    echo "  homodyne-gpu: mcmc all (GPU acceleration only)"
    echo ""
    echo "Config files in current dir:"
    local configs=(*.json(N))
    if (( ${#configs} > 0 )); then
        printf "  %s\n" "${configs[@]}"
    else
        echo "  (no .json files found)"
    fi
    echo ""
    echo "Common flags: --verbose --quiet --static-isotropic --static-anisotropic --laminar-flow"
    echo ""
    echo "GPU requirements: Linux with CUDA-enabled JAX"
}

# Try compdef registration, but don't fail if it doesn't work
# (Silent registration - no startup messages)
compdef _homodyne_complete homodyne 2>/dev/null
compdef _homodyne_gpu_complete homodyne-gpu 2>/dev/null

# For homodyne-config, compdef has issues with the dash, so use compctl as fallback
if ! compdef _homodyne_config_complete homodyne-config 2>/dev/null; then
    # Use compctl as fallback for commands with dashes
    if compctl -K _homodyne_config_complete homodyne-config 2>/dev/null; then
        # Successfully registered with compctl
        true
    else
        # If both compdef and compctl fail, provide alternative shortcuts
        echo "Note: Automatic completion for homodyne-config may not work."
        echo "Use these shortcuts instead:"
        echo "  homodyne-config --mode static_isotropic"
        echo "  homodyne-config --mode static_anisotropic"
        echo "  homodyne-config --mode laminar_flow"
    fi
fi
