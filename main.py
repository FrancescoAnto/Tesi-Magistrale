import os
import numpy as np
import matplotlib.pyplot as plt
from solvers.toy_problem.solve_toy_problem import solve_toy_problem

# ==========================================
# 1. PARAMETRI DI INPUT
# ==========================================
boundsx = [0.0, 4.0]
boundst = [0.0, 2.0]
nu = 0.2
param = 1.0
nx = 500
L = 1.0
x1 = 2*L
x2 = 3*L

# Nome del file in cui salveremo i dati
data_filename = "risultati_simulazione.npz"

# ==========================================
# 2. AVVIO DELLA SIMULAZIONE O CARICAMENTO
# ==========================================
if not os.path.exists(data_filename):
    print(">> Nessun dato salvato trovato. Inizio calcolo numerico...")
    
    # [NOTA: Assicurati di passare t_array e bc_array se usi la versione di prima!]
    final_field, storage = solve_toy_problem(boundsx, boundst, L, nu, param, nx, x1, x2)
    
    # Estraiamo i dati dal MemoryStorage
    tempi = np.array(storage.times)
    dati_u = np.array(storage.data)
    x_coords = storage[0].grid.axes_coords[0]
    
    # Salviamo tutto in un file compresso .npz
    np.savez(data_filename, tempi=tempi, dati_u=dati_u, x_coords=x_coords)
    print(f">> Calcolo completato e salvato in '{data_filename}'!")

else:
    print(f">> Trovato file '{data_filename}'. Caricamento dati in corso...")
    # Carichiamo i dati pre-calcolati
    dati_caricati = np.load(data_filename)
    tempi = dati_caricati['tempi']
    dati_u = dati_caricati['dati_u']
    x_coords = dati_caricati['x_coords']
    print(">> Dati caricati con successo! Avvio animazione...")

# ==========================================
# 3. VISUALIZZAZIONE (ANIMAZIONE)
# ==========================================
plt.ion() 
fig, ax = plt.subplots(figsize=(10, 6))

# Usiamo i dati estratti per il plot iniziale
line, = ax.plot(x_coords, dati_u[0], '-', color='b', linewidth=1, label='u(x, t)')

ax.set_xlim(boundsx[0], boundsx[1])
ax.set_xlabel('Spazio (x)')
ax.set_ylabel('Ampiezza Campo (u)')
ax.grid()

titolo_dinamico = ax.set_title('', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

y_min, y_max = dati_u.min(), dati_u.max()
margin_y = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
ax.set_ylim(y_min - margin_y, y_max + margin_y)

# Iteriamo sugli array estratti, non più sul MemoryStorage
for tempo, dati_u_t in zip(tempi, dati_u):
    line.set_ydata(dati_u_t) 
    titolo_dinamico.set_text(f'Evoluzione PDE (t={tempo:.2f} s)')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01) 

plt.ioff() 
plt.show()


