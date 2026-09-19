{
  description = "xarray-sql development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Shared libraries that manylinux wheels (pyarrow, duckdb, polars,
        # numpy, ...) expect to find via the dynamic linker at runtime.
        # NixOS doesn't put these on a standard FHS path, so we point
        # LD_LIBRARY_PATH at the nix store paths that provide them.
        runtimeLibs = with pkgs; [
          stdenv.cc.cc
          zlib
          openssl
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
            rustc
            cargo
            pkg-config
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

          # Use the Nix-provided interpreter above instead of having uv
          # download its own Python build.
          UV_PYTHON_PREFERENCE = "only-system";

          shellHook = ''
            echo "xarray-sql dev shell: $(python3 --version), $(rustc --version)"
          '';
        };
      });
}
