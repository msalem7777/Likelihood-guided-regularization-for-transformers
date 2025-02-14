def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type3':
        lr_adjust = {10: args.learning_rate * 0.5 ** 1, 20: args.learning_rate * 0.5 ** 2,
                     30: args.learning_rate * 0.5 ** 3, 40: args.learning_rate * 0.5 ** 4,
                     50: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type4':
        lr_adjust = {20: args.learning_rate * 0.5 ** 1, 40: args.learning_rate * 0.5 ** 2,
                     60: args.learning_rate * 0.5 ** 3, 80: args.learning_rate * 0.5 ** 4,
                     100: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type5':
        lr_adjust = {40: args.learning_rate * 0.5 ** 1, 80: args.learning_rate * 0.5 ** 2,
                     120: args.learning_rate * 0.5 ** 3, 160: args.learning_rate * 0.5 ** 4,
                     200: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type6':
        lr_adjust = {250: args.learning_rate * 0.5 ** 1, 500: args.learning_rate * 0.5 ** 2,
                     750: args.learning_rate * 0.5 ** 3, 1000: args.learning_rate * 0.5 ** 4,
                     2000: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type7':
        lr_adjust = {30: args.learning_rate * 0.5 ** 1, 60: args.learning_rate * 0.5 ** 2,
                     90: args.learning_rate * 0.5 ** 3, 120: args.learning_rate * 0.5 ** 4,
                     150: args.learning_rate * 0.5 ** 5,
                     180: args.learning_rate * 0.5 ** 5,
                     210: args.learning_rate * 0.5 ** 5,
                     240: args.learning_rate * 0.5 ** 5,
                     270: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))