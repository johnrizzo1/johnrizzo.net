{ pkgs, lib, config, inputs, ... }:
let 
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  languages.nix.enable = true;
  languages.javascript.enable = true;
  languages.javascript.npm.enable = true;
  languages.typescript.enable = true;
  languages.go.enable = true;

  packages = [ pkgs.git pkgs-unstable.hugo ];
  enterShell = ''
    git --version
  '';
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
  # See full reference at https://devenv.sh/reference/options/
}
