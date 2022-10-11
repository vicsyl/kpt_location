import wandb


def wand_log_me(msg, conf):
    if conf['enable_wandlog']:
        wandb.log(msg)


def log_table(data, column, table_ref=None, title=None):
    if not title:
        title = column
    if not table_ref:
        table_ref = column
    t_d = wandb.Table(data=data, columns=[column])
    wandb.log({table_ref: wandb.plot.histogram(t_d, column, title=title)})
