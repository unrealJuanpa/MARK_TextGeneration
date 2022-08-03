from MARK import MARK

m = MARK()

m.interact_text = ""

print('\nModo interactivo iniciado...\n')

while True:
  usr_input = input('USER: ').strip() + '\n'
  print('MARK: ', end='')
  m.interact(usr_input, length='\n', live_mode=True)

