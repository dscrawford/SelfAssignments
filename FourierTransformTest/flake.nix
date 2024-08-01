{
  description = "Fourier Transform Test";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem(system :
      let
        pkgs = import nixpkgs { system = system; };
        pythonPackages = ps: with ps; with pkgs; [
          numpy
          matplotlib
          scikit-learn
          jupyter
        ];
        pythonEnv = pkgs.python3.withPackages pythonPackages;
        myPackage = pkgs.stdenv.mkDerivation rec {
          pname = "fourier-transform-test";
          version = "0.1";
          
          src = ./.; # Source directory

          buildInputs = [ pythonEnv ];

          # installPhase = ''
          #   mkdir -p $out
          #   python ${src}/main.py
          #   cp -t $out plot.png model.pkl
          # '';

          meta = with pkgs.lib; {
            description = "Fourier transform test";
            license = licenses.mit;
          };
        };
      in {
        packages.default = myPackage;

        defaultPackage = myPackage;
        
        devShell = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
        };
      });
}
