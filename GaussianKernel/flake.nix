{
  description = "Gaussian Kernel Project woo";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/5ad6a14c6bf098e98800b091668718c336effc95";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem(system :
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs { system = system; };
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.pandas
            python3Packages.numpy
            python3Packages.matplotlib
            python3Packages.scikit-learn
            python3Packages.pip
          ];
        };
      });
}
