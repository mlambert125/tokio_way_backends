{
  description = "tokio_way_backends development environment flake";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix?rev=27bc56a43b7ead695d1a1d598653a4c53ff32e5d"; # - 12/5 (4)
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    fenix,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      rust = fenix.packages.${system}.complete.toolchain;
      rust-analyzer = fenix.packages.${system}.complete.rust-analyzer;
      clippy = fenix.packages.${system}.complete.clippy;
      rustfmt = fenix.packages.${system}.complete.rustfmt;
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rust
          rust-analyzer
          rustfmt
          clippy
          nixd
          alejandra
        ];
      };
    });
}
