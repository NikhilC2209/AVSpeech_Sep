def sir(target, pred):
    delta = 1e-7
    num = np.sum(np.square(target))
    den = np.sum(np.square(pred))
    num += delta
    den += delta
    return np.abs(10 * np.log10(num / den))

def sar(target, pred, enoise, eartif):
    delta = 1e-7 
    num = np.sum(np.square(target + pred + enoise), axis=(1, 2))
    den = np.sum(np.square(eartif), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

def sdr(target, pred): 
    delta = 1e-7 
    num = np.sum(np.square(target), axis=(1, 2)) 
    den = np.sum(np.square(target - pred), axis=(1, 2)) 
    num += delta 
    den += delta 
    return 10 * np.log10(num / den) 

target=np.load(r"")
pred=np.load(r"")