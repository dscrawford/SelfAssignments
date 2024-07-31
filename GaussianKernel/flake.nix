{
  description = "Gaussian Kernel Project woo";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/5ad6a14c6bf098e98800b091668718c336effc95";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem(system :
      let
        pkgs = import nixpkgs { system = system; };
        pythonPackages = ps: with ps; with pkgs; [
          pandas
          numpy
          matplotlib
          scikit-learn
          pip
        ];
        pythonEnv = pkgs.python3.withPackages pythonPackages;
        # Define a package that includes `main.py`
        myPackage = pkgs.stdenv.mkDerivation {
          pname = "gaussian-kernel-test";
          version = "1.0";
          
          src = ./.; # Source directory

          buildInputs = [ pythonEnv pkgs.bash ];

          installPhase = ''
            mkdir -p $out/bin
            cp ${myPackage.src}/main.py $out/bin/main.py
            chmod +x $out/bin/main.py
          '';

          meta = with pkgs.lib; {
            description = "Gaussian Kernel test";
            license = licenses.mit;
          };
        };
      in {
        packages.default = myPackage;

        defaultPackage = myPackage;
        
        devShell = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
        };
        
        defaultApp = {
          type = "app";
          program = "${pythonEnv}/bin/python ${myPackage}/bin/main.py";
        };
      });
}
