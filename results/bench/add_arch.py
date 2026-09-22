"""Add --arch to train_predictor.py so it can train any predictor family. Run once."""
import io

p = "proj1/scripts/train_predictor.py"
s = io.open(p, encoding="utf-8").read()

s = s.replace("from models.egnn import EGNNScalar  # noqa: E402",
              "from models.egnn import EGNNScalar  # noqa: E402\n"
              "from models.predictors import InvariantTransformer, RidgeDescriptor  # noqa: E402")

s = s.replace('''    ap.add_argument("--split", required=True, choices=["train_a", "train_b"])''',
'''    ap.add_argument("--split", required=True, choices=["train_a", "train_b"])
    ap.add_argument("--arch", default="egnn",
                    choices=["egnn", "transformer", "ridge"],
                    help="predictor family. f_A (the guide) and f_B (the evaluator) "
                         "should NOT share one: the split makes their data disjoint, "
                         "a different architecture makes their blind spots disjoint "
                         "too. See FA_FB_ARCHITECTURE_DECISION.md. 'ridge' is fitted "
                         "in closed form and ignores --epochs/--lr.")
    ap.add_argument("--heads", type=int, default=8, help="transformer only")''')

s = s.replace('''    model = EGNNScalar(len(d["types"]), args.hidden, args.layers).to(dev)''',
'''    K = len(d["types"])
    if args.arch == "egnn":
        model = EGNNScalar(K, args.hidden, args.layers).to(dev)
    elif args.arch == "transformer":
        model = InvariantTransformer(K, args.hidden, args.layers,
                                     n_heads=args.heads).to(dev)
    else:
        model = RidgeDescriptor(K).to(dev)''')

s = s.replace('''    tag = "%s_%s" % (name, args.prop)''',
'''    tag = "%s_%s" % (name, args.prop) if args.arch == "egnn" \\
        else "%s_%s_%s" % (name, args.prop, args.arch)''')

# ridge: closed-form fit, then straight to evaluation and save
s = s.replace('''    gen = torch.Generator().manual_seed(args.seed)
    best, hist, t0 = float("inf"), [], time.time()''',
'''    if args.arch == "ridge":
        # Closed-form ridge. No epochs, no LR; "training" is one linear solve,
        # which is the point of having it as the floor.
        t0 = time.time()
        yz = ((y[tr_idx] - y_mean) / y_std)
        model.fit(coords[tr_idx], feats[tr_idx], mask[tr_idx], yz, device=dev)
        mae = evaluate(va_idx)
        el = time.time() - t0
        print("  ridge fitted in %.1fs | val MAE = %.5f (%s)" % (el, mae, args.prop))
        atomic = {"state_dict": model.state_dict(), "args": vars(args),
                  "y_mean": y_mean, "y_std": y_std, "prop": args.prop,
                  "split": args.split, "val_mae": mae, "arch": args.arch,
                  "seconds": el}
        torch.save(atomic, os.path.join(CKPT_DIR, tag + ".pt"))
        json.dump({"best_val_mae": mae, "history": [], "args": vars(args),
                   "params": sum(p.numel() for p in model.parameters()),
                   "seconds": el},
                  open(os.path.join(CKPT_DIR, tag + "_history.json"), "w"), indent=2)
        print("%s done: best val MAE = %.5f (%s)  in %.0fs" % (tag, mae, args.prop, el))
        return 0

    gen = torch.Generator().manual_seed(args.seed)
    best, hist, t0 = float("inf"), [], time.time()''')

# record arch in the saved checkpoint for the SGD path too
s = s.replace('''                torch.save({"state_dict": model.state_dict(), "args": vars(args),''',
'''                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "arch": args.arch,''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("train_predictor.py: --arch {egnn,transformer,ridge} wired in")
