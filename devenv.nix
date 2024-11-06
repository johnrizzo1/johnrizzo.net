{ pkgs, lib, config, inputs, ... }:

{
  env.GREET = "jmclothingmfr.com dev environment";
  packages = [ pkgs.git pkgs.hugo ];
  enterShell = ''
    git --version
  '';
  enterTest = ''
    echo "Running tests"
    git --version | grep "2.42.0"
  '';
  languages.nix.enable = true;
  languages.javascript.enable = true;
  languages.javascript.npm.enable = true;
  languages.typescript.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
