import csv
from pathlib import Path
from xml.sax.saxutils import escape


CSV_PATH = Path('throughput.csv')
OUTPUT_PATH = Path('throughput_metrics.svg')


def load_rows(path):
    with path.open(newline='') as csv_file:
        rows = []
        for row in csv.DictReader(csv_file):
            count = float(row['count'])
            time = float(row['time'])
            rows.append({
                'count': count,
                'time': time,
                'time_per_element': time / count,
                'time_per_distance': time / (count ** 2),
            })
    return rows


def nice_number(value):
    return f'{value:.4g}'


def point_scale(values, output_min, output_max):
    data_min = min(values)
    data_max = max(values)
    if data_min == data_max:
        midpoint = (output_min + output_max) / 2
        return lambda _: midpoint
    return lambda value: output_min + (value - data_min) * (output_max - output_min) / (data_max - data_min)


def svg_text(x, y, text, *, size=13, anchor='middle', weight='normal'):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(text)}</text>'
    )


def render_chart(rows, metric, title, top, width, chart_height, margin):
    left = margin['left']
    right = width - margin['right']
    bottom = top + chart_height
    x_values = [row['count'] for row in rows]
    y_values = [row[metric] for row in rows]
    x_for = point_scale(x_values, left, right)
    y_for = point_scale(y_values, bottom, top)

    elements = [
        svg_text(width / 2, top - 24, title, size=16, weight='bold'),
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" />',
    ]

    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = top + chart_height * fraction
        value = max(y_values) - (max(y_values) - min(y_values)) * fraction
        elements.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#ddd" />')
        elements.append(svg_text(left - 10, y + 4, nice_number(value), anchor='end', size=11))

    for row in rows:
        x = x_for(row['count'])
        elements.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom + 5}" stroke="#222" />')
        elements.append(svg_text(x, bottom + 20, nice_number(row['count']), size=11))

    points = [(x_for(row['count']), y_for(row[metric])) for row in rows]
    point_string = ' '.join(f'{x},{y}' for x, y in points)
    elements.append(f'<polyline points="{point_string}" fill="none" stroke="#276fbf" stroke-width="2.5" />')

    for x, y in points:
        elements.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#276fbf" />')

    elements.append(svg_text(width / 2, bottom + 42, 'count', size=13))
    return '\n'.join(elements)


def render_svg(rows):
    width = 1000
    chart_height = 180
    gap = 95
    margin = {'left': 115, 'right': 40}
    height = 70 + chart_height * 3 + gap * 2 + 60

    charts = [
        ('time', 'Total time'),
        ('time_per_element', 'Time per element'),
        ('time_per_distance', 'Time per distance calculation (time / count^2)'),
    ]

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        svg_text(width / 2, 36, 'Pairwise Distance Timing', size=22, weight='bold'),
    ]

    top = 80
    for metric, title in charts:
        elements.append(render_chart(rows, metric, title, top, width, chart_height, margin))
        top += chart_height + gap

    elements.append('</svg>')
    return '\n'.join(elements)


def main():
    rows = load_rows(CSV_PATH)
    OUTPUT_PATH.write_text(render_svg(rows))
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
